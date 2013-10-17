#include "Graph.h"
#include <set>
#include <map>
#include <queue>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
const long DELTA = 1000;
typedef std::pair<int64_t,int64_t> Work;
pthread_barrier_t workFinishedBarrier;
pthread_barrier_t stepDeltaBarrier;
struct WorkList {
  std::queue<Work> localList;
  std::queue<Work> nextBucketList;
  int currentBucket; 
  void addWork(int64_t node, int64_t relaxed_weight) { 
    if (relaxed_weight / DELTA == currentBucket) { 
      localList.push(std::make_pair(node,relaxed_weight));
    } else { 
      nextBucketList.push(std::make_pair(node,relaxed_weight));
    }
  }
  bool getWork(Work &work) {
    if (localList.size() == 0) {
      return false;
    }
    Work w = localList.front();
    work.first = w.first; 
    work.second = w.second;
    localList.pop();
    return true;
  }
  WorkList() { 
    currentBucket = 0;
  }
};
struct WorklistManager {
  std::vector<WorkList> localLists; 
  std::vector<bool> hasWork;
  Graph &graph;
  std::vector<int64_t> &relaxed_weights;
  std::map<int,std::set<int64_t> > globalList;
  SpinLock globalQLock;
  void setNumThreads(int numThreads) { 
    localLists.resize(numThreads);
    hasWork.resize(numThreads); 
    std::fill(hasWork.begin(),hasWork.end(),false);
  }
  WorklistManager(std::vector<int64_t> &weights,Graph &g):relaxed_weights(weights),graph(g) {
  };
  WorkList& getLocalList(int threadId) { 
    return localLists[threadId];
  }
  void addUnitGlobal(Work work) { 
    int bucket = work.second / DELTA;
    if (globalList.find(bucket) == globalList.end()) { 
      std::set<int64_t> newSet; 
      newSet.insert(work.first);
      globalList[bucket] = newSet;
      return;
    }
    globalList[bucket].insert(work.first);
  }
  void transferWork(int threadId) { 
    globalQLock.Lock(); 
    //std::cout<<"Transfering work "<<threadId<<std::endl;
    WorkList &threadList(getLocalList(threadId)); 
    while (!threadList.nextBucketList.empty()) { 
      Work work = threadList.nextBucketList.front();
      threadList.nextBucketList.pop();
      addUnitGlobal(work);
      //printf("%d %d\n",work.first,work.second);
    }
   // std::cout.flush();
    globalQLock.UnLock();
  }
  bool hasMoreWork(int bucket) { 
    bool moreWork = false;
    for (int i = 0;i < hasWork.size(); i++) { 
      moreWork |= hasWork[i];
    }
    if (globalList.upper_bound(bucket) != globalList.end()) {
      moreWork = true;
    }
    return moreWork;
  }
  void getWorkFromGlobal(int threadId, int bucket) { 
    //globalQ.Lock();
    if ((globalList.find(bucket) == globalList.end()) || (globalList[bucket].size() ==0)) {
      //globalQ.unLock();
      return;
    }
    WorkList &threadList(getLocalList(threadId));
    std::set<int64_t> &workSet(globalList[bucket]);
    int numThreads = localLists.size();
    int work_size = (workSet.size() + numThreads) / numThreads; 
    int start = threadId * work_size;
    int end = (threadId + 1) * work_size;
    if (end >= workSet.size()) { 
      end = workSet.size();
    }
    int count = 0; 
    std::set<int64_t>::iterator it = workSet.begin(); 
    while(it != workSet.end()) { 
      if (count >=start && count<end) { 
        hasWork[threadId] = true;
        //std::cout<<"Thread "<<threadId<<" pushing "<<*it;
        threadList.addWork(*it,relaxed_weights[*it]);
      }
      it++;
      count++;
    }
    //cleanGlobalQ(bucket);
  }
  void cleanGlobalQ(int bucket) {
    globalQLock.Lock();
    if ((globalList.find(bucket) == globalList.end()) || (globalList[bucket].size() ==0)) {
      globalQLock.UnLock();
      return;
    }
    globalList.erase(globalList.find(bucket));
    globalQLock.UnLock();
  }
};
struct ThreadLoad {
  WorklistManager *wlManager;
  int id;
  ThreadLoad(WorklistManager *manager,int id) {
   this->wlManager = manager; 
   this->id = id;
  }
  ThreadLoad() { 
  }
};
void* threadOperator(void *arg) {
  ThreadLoad *threadLoad = (ThreadLoad*)arg; 
  WorklistManager *manager = threadLoad->wlManager; 
  int id = threadLoad->id;
  std::vector<int64_t> &relaxed_weights(manager->relaxed_weights);
  Graph &graph(manager->graph);
  WorkList &localWorkList(manager->getLocalList(id));
  //check the local work list 
  int bucket = -1;
  localWorkList.currentBucket = bucket;
INIT:
  Work work;
  while(localWorkList.getWork(work)) { 
    int64_t source = work.first; 
    int64_t weight = work.second;
    if ( relaxed_weights[source] < weight) 
      continue; 
    relaxed_weights[source] = weight; 
    std::vector<Edge> &edges(graph.getEdges(source));
    for (int i = 0; i < edges.size(); i++)  {
      int dest = edges[i].first; 
      int edgeWeight = edges[i].second; 
      //lock
      if (relaxed_weights[dest] > relaxed_weights[source] + edgeWeight) { 
        relaxed_weights[dest] = relaxed_weights[source] + edgeWeight;
        localWorkList.addWork(dest,relaxed_weights[dest]);
      }
    }
  }
  /* 
   * 1. Thread is out of work in its local work list. 
   * 2. Reach a barrier, on which you put all your work in the global list
   * 3. Reach a barrier, on which you get the work from the global list for the next bucket
   * 4. If there is no work for the next bucket check if everybody doesn't have work .
   *    you go back to executing and wait for other thread
   *    to finish the current bucket. And then do the same thing
   */
  // std::cout<<"Transfering work";
   manager->transferWork(id);
   int rc = pthread_barrier_wait(&workFinishedBarrier);
   if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
     printf("ERROR");
   }
   manager->hasWork[id] = false;
   bucket+=1;
   //printf("Incremented Bucket %d %d\n",id,bucket);
   localWorkList.currentBucket = bucket; 
   manager->getWorkFromGlobal(id,bucket);
   rc = pthread_barrier_wait(&stepDeltaBarrier);
   if(manager->hasMoreWork(bucket)) 
    goto INIT;
   printf("\n Exiting %d",id);
   pthread_exit(arg);
}
struct Chaotic {
  void operator()(Graph &graph,int64_t source, std::vector<int64_t>& relaxed_weights) {
    relaxed_weights[source] = 0; 
    int numThreads = SSSP::numThreads;
    WorklistManager manager(relaxed_weights,graph);
    manager.setNumThreads(numThreads);
    manager.globalList[0].insert(0);
    std::cout<<"Executing Chaotic at "<<numThreads<<" threads"<<std::endl;
    if(pthread_barrier_init(&workFinishedBarrier, NULL, numThreads)) {
      printf("Could not create a barrier\n");
      return;
    }
    if(pthread_barrier_init(&stepDeltaBarrier, NULL, numThreads)) {
      printf("Could not create a barrier\n");
      return;
    }
    pthread_t *threads;
    pthread_attr_t pthread_custom_attr;
    pthread_attr_init(&pthread_custom_attr);
    threads=(pthread_t *)malloc(numThreads*sizeof(*threads));
    ThreadLoad* payLoad = new ThreadLoad[numThreads];
    int i = 0;
    for(i=0;i<numThreads;i++) {
      int threadId = i;
      payLoad[i].id = i; 
      payLoad[i].wlManager = &manager;
      pthread_create(&threads[i], &pthread_custom_attr, threadOperator,(void *)(payLoad + i));
    }
    for(i=0;i < numThreads;i++){
      void* temp_parm;
      pthread_join(threads[i],&temp_parm);
    }          
  }
};
