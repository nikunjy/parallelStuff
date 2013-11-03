/** Single source shortest paths -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Single source shortest paths.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "Galois/Graphs/FileGraph.h"

#include "Lonestar/CommandLine.h"

#include <pthread.h>
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <list>
#include <vector>
#include <algorithm>

//#define _DEBUG

static const char* help =
  "<input file> <startnode> <reportnode> [-delta <deltaShift>] [-bfs]";

struct Node {
  unsigned dist;
  Node() : dist(std::numeric_limits<unsigned>::max() - 1) { }
};

typedef Galois::Graph::LC_FileGraph<Node, unsigned> Graph;
typedef Graph::GraphNode GNode;

class Synchronizer {
  struct ReducerData {
    void* data __attribute__((aligned(64)));
  };

  typedef std::pair<GNode,unsigned> Message;
  typedef std::vector<Message> Queue;
  typedef std::vector<Queue> Queues;

  Queues queues;

  ReducerData rdata[1024];
  pthread_barrier_t b;
  pthread_barrier_t reducer1;
  pthread_barrier_t reducer2;
  unsigned num_threads;

public:
  typedef Queues::iterator QueuesIterator;
  typedef Queue::iterator QueueIterator;

  Synchronizer(unsigned n): num_threads(n) { 
    pthread_barrier_init(&b, NULL, num_threads);
    pthread_barrier_init(&reducer1, NULL, num_threads);
    pthread_barrier_init(&reducer2, NULL, num_threads);
    if (num_threads >= 1024) {
      assert(0 && "Too many theads");
      abort();
    }
    for (unsigned i = 0; i < num_threads * num_threads; ++i) {
      queues.push_back(Queue());
    }
  }
  
  QueuesIterator queues_begin(unsigned id) {
    return queues.begin() + (id * num_threads);
  }

  QueuesIterator queues_end(unsigned id) {
    return queues.begin() + (++id * num_threads);
  }

  bool isLocal(unsigned id, GNode n) {
    return (n % num_threads) == id;
  }

  void barrier() {
    int rc = pthread_barrier_wait(&b);
    
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
      assert(0 && "Problem");
      abort();
    }
  }

  void sendRequest(unsigned id, GNode u, GNode v, unsigned x) {
#ifdef _DEBUG
    std::cerr << "# " << id << "send(" << u << ", " << v << ", " << x << ")" << std::endl;
#endif
    unsigned dest = v % num_threads;
    queues[dest * num_threads + id].push_back(std::make_pair(v,x));
  }

  template<typename T, typename Function>
  T reduce(unsigned id, T item, Function f) {
    rdata[id].data = &item;
    int rc = pthread_barrier_wait(&reducer1);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
      assert(0 && "Problem");
      abort();
    }

    T result = *static_cast<T*>(rdata[0].data);

    for (unsigned i = 1; i < num_threads; ++i) {
      result = f(result, *static_cast<T*>(rdata[i].data));
    }
    rc = pthread_barrier_wait(&reducer2);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
      assert(0 && "Problem");
      abort();
    }

    return result;
  }
};

class DeltaSteppingProcess {
  typedef std::list<GNode> Bucket;
  typedef Bucket::iterator BucketIterator;
  typedef std::vector<Bucket*>::size_type BucketIndex;

  std::vector<BucketIterator> position_in_bucket;
  std::vector<Bucket*> buckets;
  std::list<GNode> dummy_list;
  std::vector<bool> vertex_was_deleted;
  Graph& g;
  Synchronizer& sync;
  unsigned delta;
  unsigned id;

public:
  DeltaSteppingProcess(Graph& _g, Synchronizer& s, unsigned d, unsigned _id): 
    g(_g), sync(s), delta(d), id(_id) { }

  ~DeltaSteppingProcess() {
    for (std::vector<Bucket*>::iterator ii = buckets.begin(), ei = buckets.end(); ii != ei; ++ii) {
      if (*ii) {
        delete *ii;
        *ii = 0;
      }
    }
  }

  void init(GNode s) {
    // None of the vertices are stored in the bucket.
    position_in_bucket.clear();
    position_in_bucket.resize(g.size(), dummy_list.end());

    // None of the vertices have been deleted
    vertex_was_deleted.clear();
    vertex_was_deleted.resize(g.size(), false);

    // No path from s to any other vertex, yet
    //for (Graph::iterator ii = g.begin(), ei = g.end();
    //    ii != ei; ++ii) {
    //  if (synch.isLocal(*ii))
    //    g.getData(*ii) = inf;
    //}

    if (sync.isLocal(id, s))
      relax(s, s, 0);
  }

  void synchronize() {
#ifdef _DEBUG
    if (id == 0) {
      std::cerr << "ENTER" << std::endl;
    }
#endif
    sync.barrier();
    for (Synchronizer::QueuesIterator ii = sync.queues_begin(id),
        ei = sync.queues_end(id); ii != ei; ++ii) {
      for (Synchronizer::QueueIterator jj = ii->begin(), ej = ii->end();
          jj != ej; ++jj) {
        relax(jj->first, jj->first, jj->second);
      }
      ii->clear();
    }
    //sync.barrier();
#ifdef _DEBUG
    if (id == 0) {
      std::cerr << "leave" << std::endl;
    }
#endif
  }

  void run() {
    BucketIndex max_bucket = (std::numeric_limits<BucketIndex>::max)();
    BucketIndex current_bucket = 0;
    while (true) {
      // Synchronize with all of the other processes.
      synchronize();

      // Find the next bucket that has something in it.
      while (current_bucket < buckets.size() 
             && (!buckets[current_bucket] || buckets[current_bucket]->empty()))
        ++current_bucket;
      if (current_bucket >= buckets.size())
        current_bucket = max_bucket;

#if 0
      std::cerr << "#" << id << ": lowest bucket is #" 
                << current_bucket << std::endl;
#endif

      // Find the smallest bucket (over all processes) that has vertices
      // that need to be processed.
      
      // ddn: Cast is sometimes needed because some versions of gcc have
      // problems with templates:
      //  http://stackoverflow.com/questions/2861497/c-boost-function-overloaded-template
      typedef const BucketIndex& (*MinFn)(const BucketIndex&, const BucketIndex&);
      current_bucket = sync.reduce(id, current_bucket,
          static_cast<MinFn>(&std::min<BucketIndex>));

      if (current_bucket == max_bucket)
        // There are no non-empty buckets in any process; exit. 
        break;
#ifdef _DEBUG
    if (id == 0)
      std::cerr << "Processing bucket #" << current_bucket << std::endl;
#endif
      // Contains the set of vertices that have been deleted in the
      // relaxation of "light" edges. Note that we keep track of which
      // vertices were deleted with the property map
      // "vertex_was_deleted".
      std::vector<GNode> deleted_vertices;

      // Repeatedly relax light edges
      bool nonempty_bucket;
      do {
        // Someone has work to do in this bucket.

        if (current_bucket < buckets.size() && buckets[current_bucket]) {
          Bucket& bucket = *buckets[current_bucket];
          // For each element in the bucket
          while (!bucket.empty()) {
            GNode u = bucket.front();
#ifdef _DEBUG
          std::cerr << "#" << id << ": processing vertex " 
                    << u << std::endl;
#endif
            // Remove u from the front of the bucket
            bucket.pop_front();
            
            // Insert u into the set of deleted vertices, if it hasn't
            // been done already.
            if (!vertex_was_deleted[u]) {
              vertex_was_deleted[u] = true;
              deleted_vertices.push_back(u);
            }

            // Relax each light edge. 
            unsigned u_dist = g.getData(u, Galois::NONE).dist;
            for (Graph::neighbor_iterator ii = g.neighbor_begin(u, Galois::NONE),
                ei = g.neighbor_end(u, Galois::NONE); ii != ei; ++ii) {
              unsigned w = g.getEdgeData(u, *ii, Galois::NONE);
              if (w <= delta) // light edge
                relax(u, *ii, u_dist + w);
            }
          }
        }

        // Synchronize with all of the other processes.
        synchronize();

        // Is the bucket empty now?
        nonempty_bucket = (current_bucket < buckets.size() 
                           && buckets[current_bucket]
                           && !buckets[current_bucket]->empty());
#ifdef _DEBUG
      std::cerr << "#" << id << ": non-empty bucket " 
                << nonempty_bucket << std::endl;
#endif
       } while (sync.reduce(id, nonempty_bucket, std::logical_or<bool>()));

      // Relax heavy edges for each of the vertices that we previously
      // deleted.
      for (std::vector<GNode>::iterator iter = deleted_vertices.begin();
           iter != deleted_vertices.end(); ++iter) {
        // Relax each heavy edge. 
        GNode u = *iter;
        unsigned u_dist = g.getData(u, Galois::NONE).dist;
        for (Graph::neighbor_iterator ii = g.neighbor_begin(u, Galois::NONE),
            ei = g.neighbor_end(u, Galois::NONE); ii != ei; ++ii) {
          unsigned w = g.getEdgeData(u, *ii);
          if (w > delta) // heavy edge
            relax(u, *ii, u_dist + w);
        }
      }

      // Go to the next bucket: the current bucket must already be empty.
      ++current_bucket;
    } 
  }

  void relax(GNode u, GNode v, unsigned x) {
#ifdef _DEBUG
  std::cerr << "#" << id << ": relax(" 
            << u 
            << ", " 
            << v << ", "
            << x << ")" << std::endl;
#endif
    if (x < g.getData(v, Galois::NONE).dist) { 
      // We're relaxing the edge to vertex v.
      if (sync.isLocal(id, v)) {
        // Compute the new bucket index for v
        BucketIndex new_index = static_cast<BucketIndex>(x / delta);
        
        // Make sure there is enough room in the buckets data structure.
        if (new_index >= buckets.size()) buckets.resize(new_index + 1, 0);

        // Make sure that we have allocated the bucket itself.
        if (!buckets[new_index]) buckets[new_index] = new Bucket;

        if (g.getData(v, Galois::NONE).dist != std::numeric_limits<unsigned>::max() - 1
            && !vertex_was_deleted[v]) {
          // We're moving v from an old bucket into a new one. Compute
          // the old index, then splice it in.
          BucketIndex old_index 
            = static_cast<BucketIndex>(g.getData(v, Galois::NONE).dist / delta);
          buckets[new_index]->splice(buckets[new_index]->end(),
                                     *buckets[old_index],
                                     position_in_bucket[v]);
        } else {
          // We're inserting v into a bucket for the first time. Put it
          // at the end.
          buckets[new_index]->push_back(v);
        }

        // v is now at the last position in the new bucket
        position_in_bucket[v] = buckets[new_index]->end();
        --position_in_bucket[v];

        // Update predecessor and tentative distance information
        g.getData(v, Galois::NONE).dist = x;
      } else {
        sync.sendRequest(id, u, v, x);
      }
    }
  }
};

class DeltaStepping {
  typedef std::vector<DeltaSteppingProcess*> Processes;
  Processes processes;
  Synchronizer sync;
  unsigned num_threads;
  
public:
  DeltaStepping(Graph& g, GNode source, unsigned d, unsigned n):
    sync(n), num_threads(n) {

    for (unsigned i = 0; i < num_threads; ++i) {
      processes.push_back(new DeltaSteppingProcess(g, sync, d, i));
      processes.back()->init(source);
    }
  }

  ~DeltaStepping() {
    for (Processes::iterator ii = processes.begin(),
        ei = processes.end(); ii != ei; ++ii) {
      delete *ii;
    }
  }

  static void* threadBegin(void *p) {
    static_cast<DeltaSteppingProcess*>(p)->run();
    pthread_exit(NULL);
    return NULL;
  }

  void run() {
    pthread_t threads[1024];
    
    for (unsigned i = 0; i < num_threads; ++i) {
      int rc = pthread_create(&threads[i], NULL, threadBegin, processes[i]);
      if (rc) {
        assert(0 && "Couldn't create thread");
        abort();
      }
    }

    for (unsigned i = 0; i < num_threads; ++i) {
      int rc = pthread_join(threads[i], NULL);
      if (rc) {
        assert(0 && "Problem");
        abort();
      }
    }
    //pthread_exit(NULL);
  }

};

int main(int argc, char **argv) {
  std::vector<char*> args = parse_command_line(argc, argv, help);

  if (args.size() < 3) {
    std::cerr << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  
  const char* inputfile = args[0];
  unsigned int startNode = atoi(args[1]);
  unsigned int reportNode = atoi(args[2]);
  bool do_bfs = false;
  unsigned stepShift = 10;
  for (unsigned i = 3; i < args.size(); ++i) {
    if (strcmp(args[i], "-delta") == 0 && i + 1 < args.size()) {
      stepShift = atoi(args[i+1]);
      ++i;
    } else if (strcmp(args[i], "-bfs") == 0) {
      do_bfs = true;
    } else {
      std::cerr << "unknown argument, use -help for usage information\n";
      return 1;
    }
  }

  Graph g;
  
  g.structureFromFile(inputfile);
  g.emptyNodeData();
  std::cout << "Read " << g.size() << " nodes\n";
  std::cout << "Using delta-step of " << (1 << stepShift) << "\n";
  std::cout << "Using " << numThreads << " threads\n";

  DeltaStepping p(g, startNode, 1 << stepShift, numThreads);
  Galois::StatTimer T;
  T.start();
  p.run();
  T.stop();

  std::cout << reportNode << " " << g.getData(reportNode).dist << "\n";

  return 0;
}
