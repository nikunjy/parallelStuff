/** Breadth-first search -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * Example breadth-first search application for demoing Galois system. For optimized
 * version, use SSSP application with BFS option instead.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Accumulator.h"
#include "Galois/Timer.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#ifdef GALOIS_USE_EXP
#include "Galois/PriorityScheduling.h"
#include "Galois/Runtime/ParallelWorkInline.h"
#endif
#ifdef GALOIS_USE_TBB
#include "tbb/parallel_for.h"
#include "tbb/parallel_for_each.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/concurrent_vector.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/enumerable_thread_specific.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "Lonestar/BoilerPlate.h"

#include <string>
#include <deque>
#include <sstream>
#include <limits>
#include <iostream>
#include <deque>

static const char* name = "Breadth-first Search Example";
static const char* desc =
  "Computes the shortest path from a source node to all nodes in a directed "
  "graph using a modified Bellman-Ford algorithm\n";
static const char* url = "breadth_first_search";

//****** Command Line Options ******
enum BFSAlgo {
  serial,
  serialAsync,
  serialMin,
  parallelAsync,
  parallelBarrier,
  parallelBarrierCas,
  parallelBarrierInline,
  parallelUndirected,
  parallelTBBBarrier,
  parallelTBBAsync,
  detParallelBarrier,
  detDisjointParallelBarrier,
};

enum DetAlgo {
  nondet,
  detBase,
  detDisjoint
};

namespace cll = llvm::cl;
static cll::opt<unsigned int> startNode("startnode",
    cll::desc("Node to start search from"),
    cll::init(0));
static cll::opt<unsigned int> reportNode("reportnode",
    cll::desc("Node to report distance to"),
    cll::init(1));
static cll::opt<BFSAlgo> algo(cll::desc("Choose an algorithm:"),
    cll::values(
      clEnumVal(serial, "Serial"),
      clEnumVal(serialAsync, "Serial optimized"),
      clEnumVal(serialMin, "Serial optimized with minimal runtime"),
      clEnumVal(parallelAsync, "Parallel"),
      clEnumVal(parallelBarrier, "Parallel optimized with barrier"),
      clEnumVal(parallelBarrierCas, "Parallel optimized with barrier but using CAS"),
      clEnumVal(parallelUndirected, "Parallel specialization for undirected graphs"),
      clEnumVal(detParallelBarrier, "Deterministic parallelBarrier"),
      clEnumVal(detDisjointParallelBarrier, "Deterministic parallelBarrier with disjoint optimization"),
#ifdef GALOIS_USE_EXP
      clEnumVal(parallelBarrierInline, "Parallel optimized with inlined workset and barrier"),
#endif
#ifdef GALOIS_USE_TBB
      clEnumVal(parallelTBBAsync, "TBB"),
      clEnumVal(parallelTBBBarrier, "TBB with barrier"),
#endif
      clEnumValEnd), cll::init(parallelBarrier));
static cll::opt<std::string> filename(cll::Positional,
    cll::desc("<input file>"),
    cll::Required);

static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

//****** Work Item and Node Data Defintions ******
struct SNode {
  unsigned int dist;
  unsigned int id;
};

//! ICC + GLIB 4.6 + C++0X (at least) has a weird implementation of std::pair
//#if defined(__INTEL_COMPILER) && __INTEL_COMPILER <= 1210
#if __INTEL_COMPILER
template<typename A,typename B>
struct Pair {
  A first;
  B second;
  Pair(const A& a, const B& b): first(a), second(b) { }
  Pair<A,B>& operator=(const Pair<A,B>& other) {
    if (this != &other) {
      first = other.first;
      second = other.second;
    }
    return *this;
  }
};
#else
template<typename A,typename B>
struct Pair: std::pair<A,B> { 
  Pair(const A& a, const B& b): std::pair<A,B>(a, b) { }
};
#endif

typedef Galois::Graph::LC_CSR_Graph<SNode, void> Graph;
typedef Graph::GraphNode GNode;

Graph graph;

struct UpdateRequest {
  GNode n;
  unsigned int w;

  UpdateRequest(): w(0) { }
  UpdateRequest(const GNode& N, unsigned int W): n(N), w(W) { }
  bool operator<(const UpdateRequest& o) const {
    if (w < o.w) return true;
    if (w > o.w) return false;
    return n < o.n;
  }
  bool operator>(const UpdateRequest& o) const {
    if (w > o.w) return true;
    if (w < o.w) return false;
    return n > o.n;
  }
  unsigned getID() const { return /* graph.getData(n).id; */ 0; }
};

std::ostream& operator<<(std::ostream& out, const SNode& n) {
  out <<  "(dist: " << n.dist << ")";
  return out;
}

struct UpdateRequestIndexer: public std::unary_function<UpdateRequest,unsigned int> {
  unsigned int operator()(const UpdateRequest& val) const {
    unsigned int t = val.w;
    return t;
  }
};

struct GNodeIndexer: public std::unary_function<GNode,unsigned int> {
  unsigned int operator()(const GNode& val) const {
    return graph.getData(val, Galois::NONE).dist;
  }
};

struct not_consistent {
  bool operator()(GNode n) const {
    unsigned int dist = graph.getData(n).dist;
    for (Graph::edge_iterator 
	   ii = graph.edge_begin(n),
	   ee = graph.edge_end(n); ii != ee; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      unsigned int ddist = graph.getData(dst).dist;
      if (ddist > dist + 1) {
        std::cerr << "bad level value: " << ddist << " > " << (dist + 1) << " " << n << " -> " << *ii << "\n";
	return true;
      }
    }
    return false;
  }
};

struct not_visited {
  bool operator()(GNode n) const {
    unsigned int dist = graph.getData(n).dist;
    if (dist >= DIST_INFINITY) {
      std::cerr << "unvisited node: " << dist << " >= INFINITY\n";
      return true;
    }
    return false;
  }
};

struct max_dist {
  Galois::GReduceMax<unsigned int>& m;
  max_dist(Galois::GReduceMax<unsigned int>& _m): m(_m) { }

  void operator()(GNode n) const {
    m.update(graph.getData(n).dist);
  }
};

//! Simple verifier
static bool verify(GNode source) {
  if (graph.getData(source).dist != 0) {
    std::cerr << "source has non-zero dist value\n";
    return false;
  }
  
  bool okay = Galois::ParallelSTL::find_if(graph.begin(), graph.end(), not_consistent()) == graph.end()
    && Galois::ParallelSTL::find_if(graph.begin(), graph.end(), not_visited()) == graph.end();

  if (okay) {
    Galois::GReduceMax<unsigned int> m;
    Galois::do_all(graph.begin(), graph.end(), max_dist(m));
    std::cout << "max dist: " << m.reduce() << "\n";
  }
  
  return okay;
}

static void readGraph(GNode& source, GNode& report) {
  graph.structureFromFile(filename);

  source = *graph.begin();
  report = *graph.begin();

  std::cout << "Read " << graph.size() << " nodes\n";
  
  size_t id = 0;
  bool foundReport = false;
  bool foundSource = false;
  for (Graph::iterator src = graph.begin(), ee =
      graph.end(); src != ee; ++src, ++id) {
    SNode& node = graph.getData(*src, Galois::NONE);
    node.dist = DIST_INFINITY;
    node.id = id;
    if (id == startNode) {
      source = *src;
      foundSource = true;
    } 
    if (id == reportNode) {
      foundReport = true;
      report = *src;
    }
  }

  if (!foundReport || !foundSource) {
    std::cerr 
      << "failed to set report: " << reportNode 
      << "or failed to set source: " << startNode << "\n";
    assert(0);
    abort();
  }
}

//! Serial BFS using optimized flags 
struct SerialAsyncAlgo {
  std::string name() const { return "Serial (Async)"; }

  void operator()(const GNode source) const {
    std::deque<GNode> wl;
    graph.getData(source, Galois::NONE).dist = 0;

    for (Graph::edge_iterator ii = graph.edge_begin(source, Galois::NONE), 
           ei = graph.edge_end(source, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);
      ddata.dist = 1;
      wl.push_back(dst);
    }

    while (!wl.empty()) {
      GNode n = wl.front();
      wl.pop_front();

      SNode& data = graph.getData(n, Galois::NONE);

      unsigned int newDist = data.dist + 1;

      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
            ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::NONE);

        if (newDist < ddata.dist) {
          ddata.dist = newDist;
          wl.push_back(dst);
        }
      }
    }
  }
};

//! Galois BFS using optimized flags
struct AsyncAlgo {
  typedef int tt_does_not_need_aborts;

  std::string name() const { return "Parallel (Async)"; }

  void operator()(const GNode& source) const {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<GNodeIndexer,dChunk> OBIM;
    
    std::deque<GNode> initial;
    graph.getData(source).dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(source),
          ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      initial.push_back(dst);
    }

    Galois::for_each<OBIM>(initial.begin(), initial.end(), *this);
  }

  void operator()(GNode& n, Galois::UserContext<GNode>& ctx) const {
    SNode& data = graph.getData(n, Galois::NONE);

    unsigned int newDist = data.dist + 1;

    for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
          ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);

      unsigned int oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= newDist)
          break;
        if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
          ctx.push(dst);
          break;
        }
      }
    }
  }
};

/**
 * Alternate between processing outgoing edges or incoming edges. Works for
 * directed graphs as well, but just implement assuming graph is symmetric so
 * we don't have to distinguish incoming or outgoing edges.
 *
 * S. Beamer, K. Asanovic and D. Patterson. Direction-optimizing breadth-first
 * search. In Supercomputing. 2012.
 */
struct UndirectedAlgo {
  std::string name() const { return "Undirected"; }
  struct CountingBag {
    Galois::InsertBag<GNode> wl;
    Galois::GAccumulator<size_t> count;

    void clear() {
      wl.clear();
      count.reset();
    }
    bool empty() {
      return wl.empty();
    }
    size_t size() {
      return count.reduce();
    }
  };

  CountingBag bags[2];

  struct ForwardProcess {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;

    UndirectedAlgo* self;
    CountingBag* next;
    unsigned int newDist;
    ForwardProcess(UndirectedAlgo* s, CountingBag* n, int d): self(s), next(n), newDist(d) { }

    void operator()(const GNode& n, Galois::UserContext<GNode>&) {
      (*this)(n);
    }

    void operator()(const Graph::edge_iterator& it, Galois::UserContext<Graph::edge_iterator>&) {
      (*this)(it);
    }

    void operator()(const Graph::edge_iterator& ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);

      unsigned int oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= newDist)
          return;
        if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
          next->wl.push(dst);
          next->count += 1
            + std::distance(graph.edge_begin(dst, Galois::MethodFlag::NONE),
              graph.edge_end(dst, Galois::MethodFlag::NONE));
        }
      }
    }

    void operator()(const GNode& n) {
      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::MethodFlag::NONE),
            ei = graph.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        (*this)(ii);
      }
    }
  };

  struct BackwardProcess {
    typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;

    UndirectedAlgo* self;
    CountingBag* next;
    unsigned int newDist; 
    BackwardProcess(UndirectedAlgo* s, CountingBag* n, int d): self(s), next(n), newDist(d) { }

    void operator()(const GNode& n, Galois::UserContext<GNode>&) {
      (*this)(n);
    }

    void operator()(const GNode& n) {
      SNode& sdata = graph.getData(n, Galois::MethodFlag::NONE);
      if (sdata.dist <= newDist)
        return;

      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::MethodFlag::NONE),
            ei = graph.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::MethodFlag::NONE);

        if (ddata.dist + 1 == newDist) {
          sdata.dist = newDist;
          next->wl.push(n);
          next->count += 1
            + std::distance(graph.edge_begin(n, Galois::MethodFlag::NONE),
              graph.edge_end(n, Galois::MethodFlag::NONE));
          break;
        }
      }
    }
  };

  void operator()(const GNode& source) {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedLIFO<256> WL;
    int next = 0;
    unsigned int newDist = 0;
    graph.getData(source).dist = 0;
    bags[next].wl.push(source);
    bags[next].count += 1 + std::distance(graph.edge_begin(source), graph.edge_end(source));
    //Galois::for_each(graph.out_edges(source, Galois::MethodFlag::NONE).begin(), 
    //    graph.out_edges(source, Galois::MethodFlag::NONE).end(),
    //    ForwardProcess(this, &bags[next], newDist));
    size_t total = 0;
    while (!bags[next].empty()) {
      size_t nextSize = bags[next].size();
      total += nextSize;
      int cur = next;
      next = (cur + 1) & 1;
      newDist++;
      if (nextSize > graph.sizeEdges() / 21)
        Galois::do_all_local(graph, BackwardProcess(this, &bags[next], newDist));
      else
        Galois::for_each<WL>(bags[cur].wl.begin(), bags[cur].wl.end(), ForwardProcess(this, &bags[next], newDist));
      bags[cur].clear();
    }
  }
};

//! BFS using optimized flags and barrier scheduling 
template<typename WL,bool useCas>
struct BarrierAlgo {
  typedef int tt_does_not_need_aborts;

  std::string name() const { return "Parallel (Barrier)"; }
  typedef Pair<GNode,int> ItemTy;

  void operator()(const GNode& source) const {
    std::deque<ItemTy> initial;

    graph.getData(source).dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(source),
          ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      initial.push_back(ItemTy(dst, 2));
    }
    Galois::for_each<WL>(initial.begin(), initial.end(), *this);
  }

  void operator()(const ItemTy& item, Galois::UserContext<ItemTy>& ctx) const {
    GNode n = item.first;

    unsigned int newDist = item.second;

    for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
          ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::NONE);

      unsigned int oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= newDist)
          break;
        if (!useCas || __sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
          if (!useCas)
            ddata.dist = newDist;
          ctx.push(ItemTy(dst, newDist + 1));
          break;
        }
      }
    }
  }
};

//! BFS using optimized flags and barrier scheduling 
template<DetAlgo Version>
struct DetBarrierAlgo {
  typedef int tt_needs_per_iter_alloc; // For LocalState

  std::string name() const { return "Parallel (Deterministic Barrier)"; }
  typedef Pair<GNode,int> ItemTy;

  struct LocalState {
    typedef std::deque<GNode,Galois::PerIterAllocTy> Pending;
    Pending pending;
    LocalState(DetBarrierAlgo<Version>& self, Galois::PerIterAllocTy& alloc): pending(alloc) { }
  };

  struct IdFn {
    unsigned long operator()(const ItemTy& item) const {
      return graph.getData(item.first, Galois::NONE).id;
    }
  };

  void operator()(const GNode& source) const {
#ifdef GALOIS_USE_EXP
    typedef GaloisRuntime::WorkList::BulkSynchronousInline<> WL;
#else
  typedef GaloisRuntime::WorkList::BulkSynchronous<GaloisRuntime::WorkList::dChunkedLIFO<256> > WL;
#endif
    std::deque<ItemTy> initial;

    graph.getData(source).dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(source),
          ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      initial.push_back(ItemTy(dst, 2));
    }
    switch (Version) {
      case nondet: 
        Galois::for_each<WL>(initial.begin(), initial.end(), *this); break;
      case detBase:
        Galois::for_each_det(initial.begin(), initial.end(), *this); break;
      case detDisjoint:
        Galois::for_each_det(initial.begin(), initial.end(), *this); break;
      default: std::cerr << "Unknown algorithm " << Version << "\n"; abort();
    }
  }

  void build(const ItemTy& item, typename LocalState::Pending* pending) const {
    GNode n = item.first;

    unsigned int newDist = item.second;
    
    for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
          ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst, Galois::ALL);

      unsigned int oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= newDist)
          break;
        pending->push_back(dst);
        break;
      }
    }
  }

  void modify(const ItemTy& item, Galois::UserContext<ItemTy>& ctx, typename LocalState::Pending* ppending) const {
    unsigned int newDist = item.second;
    bool useCas = false;

    for (typename LocalState::Pending::iterator ii = ppending->begin(), ei = ppending->end(); ii != ei; ++ii) {
      GNode dst = *ii;
      SNode& ddata = graph.getData(dst, Galois::NONE);

      unsigned int oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= newDist)
          break;
        if (!useCas || __sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
          if (!useCas)
            ddata.dist = newDist;
          ctx.push(ItemTy(dst, newDist + 1));
          break;
        }
      }
    }
  }

  void operator()(const ItemTy& item, Galois::UserContext<ItemTy>& ctx) const {
    typename LocalState::Pending* ppending;
    if (Version == detDisjoint) {
      bool used;
      LocalState* localState = (LocalState*) ctx.getLocalState(used);
      ppending = &localState->pending;
      if (used) {
        modify(item, ctx, ppending);
        return;
      }
    }
    if (Version == detDisjoint) {
      build(item, ppending);
    } else {
      typename LocalState::Pending pending(ctx.getPerIterAlloc());
      build(item, &pending);
      graph.getData(item.first, Galois::WRITE); // Failsafe point
      modify(item, ctx, &pending);
    }
  }
};

#ifdef GALOIS_USE_TBB
//! TBB version based off of AsyncAlgo
struct TBBAsyncAlgo {
  std::string name() const { return "Parallel (TBB)"; }

  struct Fn {
    void operator()(const GNode& n, tbb::parallel_do_feeder<GNode>& feeder) const {
      SNode& data = graph.getData(n, Galois::NONE);

      unsigned int newDist = data.dist + 1;

      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
            ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::NONE);

        unsigned int oldDist;
        while (true) {
          oldDist = ddata.dist;
          if (oldDist <= newDist)
            break;
          if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
            feeder.add(dst);
            break;
          }
        }
      }
    }
  };

  void operator()(const GNode& source) const {
    tbb::task_scheduler_init init(numThreads);
    
    std::vector<GNode> initial;
    graph.getData(source).dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(source),
          ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      initial.push_back(dst);
    }

    tbb::parallel_do(initial.begin(), initial.end(), Fn());
  }
};

//! TBB version based off of BarrierAlgo
struct TBBBarrierAlgo {
  std::string name() const { return "Parallel (TBB Barrier)"; }
  typedef tbb::enumerable_thread_specific<std::vector<GNode> > ContainerTy;
  //typedef tbb::concurrent_vector<GNode,tbb::cache_aligned_allocator<GNode> > ContainerTy;

  struct Fn {
    ContainerTy& wl;
    unsigned int newDist;
    Fn(ContainerTy& w, unsigned int d): wl(w), newDist(d) { }

    void operator()(const GNode& n) const {
      for (Graph::edge_iterator ii = graph.edge_begin(n, Galois::NONE),
            ei = graph.edge_end(n, Galois::NONE); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        SNode& ddata = graph.getData(dst, Galois::NONE);

        // Racy but okay
        if (ddata.dist <= newDist)
          continue;
        ddata.dist = newDist;
        //wl.push_back(dst);
        wl.local().push_back(dst);
      }
    }
  };

  struct Clear {
    ContainerTy& wl;
    Clear(ContainerTy& w): wl(w) { }
    template<typename Range>
    void operator()(const Range&) const {
      wl.local().clear();
    }
  };

  struct Initialize {
    ContainerTy& wl;
    Initialize(ContainerTy& w): wl(w) { }
    template<typename Range>
    void operator()(const Range&) const {
      wl.local().reserve(graph.size() / numThreads);
    }
  };

  void operator()(const GNode& source) const {
    tbb::task_scheduler_init init(numThreads);
    
    ContainerTy wls[2];
    unsigned round = 0;

    tbb::parallel_for(tbb::blocked_range<unsigned>(0, numThreads, 1), Initialize(wls[round]));

    graph.getData(source).dist = 0;
    for (Graph::edge_iterator ii = graph.edge_begin(source),
          ei = graph.edge_end(source); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
      SNode& ddata = graph.getData(dst);
      ddata.dist = 1;
      //wls[round].push_back(dst);
      wls[round].local().push_back(dst);
    }

    unsigned int newDist = 2;

    Galois::StatTimer Tparallel("ParallelTime");
    Tparallel.start();
    while (true) {
      unsigned cur = round & 1;
      unsigned next = (round + 1) & 1;
      //tbb::parallel_for_each(wls[round].begin(), wls[round].end(), Fn(wls[next], newDist));
      tbb::flattened2d<ContainerTy> flatView = tbb::flatten2d(wls[cur]);
      tbb::parallel_for_each(flatView.begin(), flatView.end(), Fn(wls[next], newDist));
      tbb::parallel_for(tbb::blocked_range<unsigned>(0, numThreads, 1), Clear(wls[cur]));
      //wls[cur].clear();

      ++newDist;
      ++round;
      //if (next_wl.begin() == next_wl.end())
      tbb::flattened2d<ContainerTy> flatViewNext = tbb::flatten2d(wls[next]);
      if (flatViewNext.begin() == flatViewNext.end())
        break;
    }
    Tparallel.stop();
  }
};
#else
struct TBBAsyncAlgo {
  std::string name() const { return "Parallel (TBB)"; }
  void operator()(const GNode& source) const { abort(); }
};

struct TBBBarrierAlgo {
  std::string name() const { return "Parallel (TBB Barrier)"; }
  void operator()(const GNode& source) const { abort(); }
};
#endif

template<typename AlgoTy>
void run() {
  AlgoTy algo;
  GNode source, report;
  readGraph(source, report);
  Galois::preAlloc((numThreads + (graph.size() * sizeof(SNode) * 2) / GaloisRuntime::MM::pageSize)*8);
  Galois::Statistic("MeminfoPre", GaloisRuntime::MM::pageAllocInfo());

  Galois::StatTimer T;
  std::cout << "Running " << algo.name() << " version\n";
  T.start();
  algo(source);
  T.stop();
  
  Galois::Statistic("MeminfoPost", GaloisRuntime::MM::pageAllocInfo());

  std::cout << "Report node: " << reportNode << " " << graph.getData(report) << "\n";

  if (!skipVerify) {
    if (verify(source)) {
      std::cout << "Verification successful.\n";
    } else {
      std::cerr << "Verification failed.\n";
      assert(0 && "Verification failed");
      abort();
    }
  }
}

int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  using namespace GaloisRuntime::WorkList;
  typedef BulkSynchronous<dChunkedLIFO<256> > BSWL;

#ifdef GALOIS_USE_EXP
  typedef BulkSynchronousInline<> BSInline;
#else
  typedef BSWL BSInline;
#endif

  switch (algo) {
    case serialAsync: run<SerialAsyncAlgo>(); break;
    case serialMin: run<BarrierAlgo<FIFO<int,false>,false> >(); break;
    case parallelAsync: run<AsyncAlgo>();  break;
    case parallelBarrierCas: run<BarrierAlgo<BSWL,true> >(); break;
    case parallelBarrier: run<BarrierAlgo<BSWL,false> >(); break;
    case parallelBarrierInline: run<BarrierAlgo<BSInline,false> >(); break;
    case parallelUndirected: run<UndirectedAlgo>(); break;
    case parallelTBBAsync: run<TBBAsyncAlgo>(); break;
    case parallelTBBBarrier: run<TBBBarrierAlgo>(); break;
    case detParallelBarrier: run<DetBarrierAlgo<detBase> >(); break;
    case detDisjointParallelBarrier: run<DetBarrierAlgo<detDisjoint> >(); break;
    default: std::cerr << "Unknown algorithm " << algo << "\n"; abort();
  }

  return 0;
}
