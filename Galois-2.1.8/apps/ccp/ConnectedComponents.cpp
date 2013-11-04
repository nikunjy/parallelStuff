/** Connected Components application -*- C++ -*-
 * @file
 *
 * A simple spanning tree algorithm to demostrate the Galois system.
 *
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
 * shall University be liable for incomponentental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Shweta Gulati shweta@cs.utexas.edu
 */
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/ParallelWorkInline.h"
#endif
#include "llvm/Support/CommandLine.h"
#include <set>
#include "Lonestar/BoilerPlate.h"

#include "boost/optional.hpp"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <list>
using namespace std;
const char* name = "Connected Components Problem";
const char* desc = "Find out the number of components";
const char* url = "ccp";

enum Algo {
  serial,
  parallel,
};

enum Schedule {
  FIFO,
  CHUNKED,
  DCHUNKED,
  ORDERED,
  LOCALQ,
};

namespace cll = llvm::cl;
static cll::opt<string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo(cll::desc("Serial/Parallel:"),
    cll::values(
      clEnumVal(serial, "Serial"),
      clEnumVal(parallel, "Parallel"),
      clEnumValEnd), cll::init(parallel));

static cll::opt<Schedule> schedule(cll::desc("Choose a schedule:"),
    cll::values(
      clEnumVal(FIFO, "fifo"),
      clEnumVal(CHUNKED, "chunked fifo"),
      clEnumVal(DCHUNKED, "dChunked fifo"),
      clEnumVal(ORDERED, "OBIM"),
      clEnumVal(LOCALQ, "localQ chunked fifo"),
      clEnumValEnd), cll::init(FIFO));
struct Node {
  unsigned int vertexId;
  unsigned int component;
  Node() {}
};
typedef Galois::Graph::LC_CSR_Graph<Node,void> Graph;
typedef Graph::GraphNode GNode;
Graph graph;
struct CCP {
  void operator()(GNode node, Galois::UserContext<GNode> &lwl) { 
    auto flag = Galois::NONE;
    Node &nData = graph.getData(node);
    for (Graph::edge_iterator ei = graph.edge_begin(node,flag),ee=graph.edge_end(node,flag);
      ei != ee; ei++) {
      GNode neigh = graph.getEdgeDst(ei);
      Node &neighData = graph.getData(neigh,Galois::WRITE);
      if (neighData.component > nData.component) {
        neighData.component = nData.component;
        lwl.push(neigh);
      }
    }
  }
  void operator()(GNode node, list<GNode>& workList) {
    Node &nData = graph.getData(node);
    for (Graph::edge_iterator ei = graph.edge_begin(node),ee=graph.edge_end(node);
      ei != ee; ei++) {
      GNode neigh = graph.getEdgeDst(ei);
      Node &neighData = graph.getData(neigh);
      if (neighData.component > nData.component) { 
        neighData.component = nData.component;
        if (schedule == FIFO) {
            workList.emplace_back(neigh);
        } else {       
            workList.emplace_front(neigh);
        }
      }
    }
  }
};
struct Indexer: public unary_function<GNode, unsigned int> {
    unsigned int operator()(const GNode& val) const {
      Node &data = graph.getData(val,Galois::NONE);
      unsigned int ret = data.component;
      if (ret > 10240) {
        ret = 10240;
      }
      return ret;
    }
};
#define CHUNKSIZE 512
struct GaloisAlgo {
  void operator()() {
    switch(schedule) {
      case FIFO :
        cout<<"FIFO approach"<<endl;
        Galois::for_each<GaloisRuntime::WorkList::FIFO<GNode,true> >(graph.begin(), graph.end(), CCP(),"Connected Components");
      break;
      case CHUNKED :
        cout<<"Chunked FIFO approach"<<endl;
        Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<CHUNKSIZE> >(graph.begin(), graph.end(), CCP(),"Connected Components");
      break;
      case DCHUNKED :
        cout<<"dChunked FIFO approach"<<endl;
        Galois::for_each<GaloisRuntime::WorkList::dChunkedFIFO<CHUNKSIZE> >(graph.begin(), graph.end(), CCP(),"Connected Components");
      break;
      case ORDERED :
        cout<<"OBIM approach"<<endl;
        typedef GaloisRuntime::WorkList::dChunkedFIFO<> Chunk;
        Galois::for_each<GaloisRuntime::WorkList::OrderedByIntegerMetric<Indexer,Chunk> >(graph.begin(), graph.end(), CCP(), "Connected Components");
      break;
      case LOCALQ:
        cout<<"Local Queues approach"<<endl;
        Galois::for_each<GaloisRuntime::WorkList::LocalQueues<Chunk>>(graph.begin(), graph.end(), CCP(),"Connected Components");
      break;
    }
  }
};
struct SerialAlgo {
  void operator()() {
    list<GNode> workList; 
    for (Graph::iterator gb = graph.begin(),ge = graph.end();
      gb != ge; gb++) {
      workList.emplace_back(*gb);
    }
    while (!workList.empty()) {
      GNode node = workList.front(); 
      workList.pop_front();
      CCP()(node,workList);
    }
  }
};

int main(int argc, char** argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);
  graph.structureFromFile(filename);
  unsigned int id = 0;
  for (Graph::iterator gb = graph.begin(), ge = graph.end(); gb != ge; ++gb, ++id) {
    graph.getData(*gb).vertexId = graph.getData(*gb).component = id;
  }
  Galois::StatTimer T;
  T.start();
  switch (algo) {
    case serial: SerialAlgo()(); break;
    default: GaloisAlgo()(); break;
  }
  T.stop();
  cout<<"Time Spent "<<T.get()<<" ms "<<endl;
  return 0;
}
