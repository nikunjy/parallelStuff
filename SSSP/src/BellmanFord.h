#ifndef BELLMANFORD_H
#define BELLMANFORD_H
#include "Graph.h"
#include <omp.h>
struct SpinLock { 
  int num; 
  SpinLock() { 
    num = 0;
  }
  void Lock(){
    while(!__sync_bool_compare_and_swap(&num,0,1));
  }
  void UnLock(){
    num=0;
  }
};
struct BellmanFord {
  void operator()(Graph &graph,int64_t source, std::vector<int64_t>& relaxed_weights) {
    relaxed_weights[source] = 0;
    int numThreads = SSSP::numThreads;
    std::cout<<"Num Threads"<<numThreads<<std::endl;
    std::vector<int64_t> globalQ;
   // vector<SpinLock> locks(graph.getNumNodes());
    globalQ.push_back(source);
    int64_t iter = 0;
    std::vector<EdgePair> &edges(graph.getEdges());
    omp_set_num_threads(numThreads);
    int change = 0;
    for(int j = 0; j < graph.getNumNodes(); j++) {
      change = 0;
      int64_t chunkSize = edges.size()/numThreads;
      #pragma omp parallel for shared(graph,chunkSize,edges,relaxed_weights,j) schedule(static,chunkSize) reduction(+:change)
      for (int i = 0;i<edges.size(); i++) { 
        int threadId = omp_get_thread_num();
        int64_t source = edges[i].first;
        int64_t dest = edges[i].second.first;
        int64_t weight = edges[i].second.second;
       // locks[dest].Lock(); 
        if (relaxed_weights[dest] > relaxed_weights[source] + weight) { 
          change++;
          relaxed_weights[dest] = relaxed_weights[source] + weight;
        }
        //locks[dest].UnLock();
      }
      if (change == 0) 
        break;
    }
  }
};
#endif
