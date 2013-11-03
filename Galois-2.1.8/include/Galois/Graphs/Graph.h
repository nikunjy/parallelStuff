/** Basic morph graphs -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_GRAPHS_GRAPH2_H
#define GALOIS_GRAPHS_GRAPH2_H

#include <boost/functional.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/MethodFlags.h"
#include "Galois/Runtime/InsBag.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace Galois {
//! Parallel graph data structures.
namespace Graph {

namespace GraphImpl {
/**
 * Wrapper class to have a valid type on void edges
 */
template<typename NTy, typename ETy, bool Directed>
struct EdgeItem;

template<typename NTy, typename ETy>
struct EdgeItem<NTy, ETy, true> {
  typedef ETy& reference;
  
  NTy* N;
  ETy Ea;

  inline NTy*&       first()       { assert(N); return N; }
  inline NTy* const& first() const { assert(N); return N; }
  inline ETy*       second()       { return &Ea; }
  inline const ETy* second() const { return &Ea; }
  EdgeItem(NTy* n, ETy* v) : N(n) {}
  static size_t sizeOfSecond()     { return sizeof(ETy); }
};

template<typename NTy, typename ETy>
struct EdgeItem<NTy, ETy, false> {
  typedef ETy& reference;
  
  NTy* N;
  ETy* Ea;

  inline NTy*&       first()       { assert(N); return N; }
  inline NTy* const& first() const { assert(N); return N; }
  inline ETy*       second()       { return Ea; }
  inline const ETy* second() const { return Ea; }
  EdgeItem(NTy* n, ETy* v) : N(n), Ea(v) {}
  static size_t sizeOfSecond()     { return sizeof(ETy); }
};

template<typename NTy>
struct EdgeItem<NTy, void, true> {
  typedef char& reference;

  NTy* N;
  inline NTy*&       first()        { return N; }
  inline NTy* const& first()  const { return N; }
  inline char*       second() const { return static_cast<char*>(NULL); }
  inline char*       addr()   const { return second(); }
  EdgeItem(NTy* n, void* v) : N(n) {}
  static size_t sizeOfSecond()      { return 0; }
};

template<typename NTy>
struct EdgeItem<NTy, void, false> {
  typedef char& reference;

  NTy* N;
  inline NTy*&       first()        { return N; }
  inline NTy* const& first()  const { return N; }
  inline char*       second() const { return static_cast<char*>(NULL); }
  inline char*       addr()   const { return second(); }
  EdgeItem(NTy* n, void* v) : N(n) {}
  static size_t sizeOfSecond()      { return 0; }
};

template<typename ETy>
struct EdgeFactory {
  GaloisRuntime::MM::FSBGaloisAllocator<ETy> mem;
  ETy* mkEdge() {
    ETy* e = mem.allocate(1);
    mem.construct(e, ETy());
    return e;
  }
  void delEdge(ETy* e) {
    mem.destroy(e);
    mem.deallocate(e, 1);
  }
  bool mustDel() const { return true; }
};

template<>
struct EdgeFactory<void> {
  void* mkEdge() { return static_cast<void*>(NULL); }
  void delEdge(void*) {}
  bool mustDel() const { return false; }
};

} // end namespace impl

/**
 * A Graph.
 *
 * An example of use:
 * 
 * \code
 * struct Node {
 *   ... // Definition of node data
 * };
 *
 * typedef Galois::Graph::FastGraph<Node,int,true> Graph;
 * 
 * // Create graph
 * Graph g;
 * Node n1, n2;
 * Graph::GraphNode a, b;
 * a = g.createNode(n1);
 * g.addNode(a);
 * b = g.createNode(n2);
 * g.addNode(b);
 * g.getEdgeData(g.addEdge(a, b)) = 5;
 *
 * // Traverse graph
 * for (Graph::iterator ii = g.begin(), ei = g.end(); ii != ei; ++ii) {
 *   Graph::GraphNode src = *ii;
 *   for (Graph::edge_iterator jj = g.edge_begin(src), ej = g.edge_end(src); ++jj) {
 *     Graph::GraphNode dst = graph.getEdgeDst(jj);
 *     int edgeData = g.getEdgeData(jj);
 *     assert(edgeData == 5);
 *   }
 * }
 * \endcode
 *
 * And in C++11:
 *
 * \code
 * // Traverse graph
 * for (Graph::GraphNode src : g) {
 *   for (Graph::edge_iterator edge : g.out_edges(src)) {
 *     Graph::GraphNode dst = g.getEdgeDst(edge);
 *     int edgeData = g.getEdgeData(edge);
 *     assert(edgeData == 5);
 *   }
 * }
 * \endcode
 *
 * @tparam NodeTy Type of node data
 * @tparam EdgeTy Type of edge data
 * @tparam Directional true if graph is directed
 */
template<typename NodeTy, typename EdgeTy, bool Directional>
class FirstGraph : private boost::noncopyable {
  template<typename T>
  struct first_eq_and_valid {
    T N2;
    first_eq_and_valid(T& n) :N2(n) {}
    template <typename T2>
    bool operator()(const T2& ii) const { 
      return ii.first() == N2 && ii.first() && ii.first()->active;
    }
  };
  struct first_not_valid {
    template <typename T2>
    bool operator()(const T2& ii) const { return !ii.first() || !ii.first()->active; }
  };
  
  struct gNode: public GaloisRuntime::Lockable {
    //! The storage type for an edge
    typedef GraphImpl::EdgeItem<gNode, EdgeTy, Directional> EITy;
    
    //! The storage type for edges
    typedef llvm::SmallVector<EITy, 3> EdgesTy;
    
    typedef typename EdgesTy::iterator iterator;
    
    EdgesTy edges;
    NodeTy data;
    bool active;
    
    gNode(const NodeTy& d) :data(d), active(false) { }
    gNode() :active(false) { }
    
    iterator begin() { return edges.begin(); }
    iterator end()   { return edges.end();  }
    
    void erase(iterator ii) {
      *ii = edges.back();
      edges.pop_back();
    }

    void erase(gNode* N) { 
      iterator ii = find(N);
      if (ii != end())
        edges.erase(ii); 
    }

    iterator find(gNode* N) {
      return std::find_if(begin(), end(), first_eq_and_valid<gNode*>(N));
    }

    iterator createEdge(gNode* N, EdgeTy* v) {
      //First check for holes
      iterator ii = std::find_if(begin(), end(), first_not_valid());
      if (ii != end()) {
	*ii = EITy(N, v);
	return ii;
      }
      return edges.insert(edges.end(), EITy(N, v));
    }
  };

  //The graph manages the lifetimes of the data in the nodes and edges
  typedef GaloisRuntime::galois_insert_bag<gNode> NodeListTy;
  NodeListTy nodes;

  GraphImpl::EdgeFactory<EdgeTy> edges;

  //Helpers for iterator classes
  struct is_node : public std::unary_function<gNode&, bool>{
    bool operator() (const gNode& g) const { return g.active; }
  };
  struct is_edge : public std::unary_function<typename gNode::EITy&, bool> {
    bool operator()(typename gNode::EITy& e) const { return e.first()->active; }
  };
  struct makeGraphNode: public std::unary_function<gNode&, gNode*> {
    gNode* operator()(gNode& data) const { return &data; }
  };

public:
  //! Graph node handle
  typedef gNode* GraphNode;
  //! Edge data type
  typedef EdgeTy edge_type;
  //! Node data type
  typedef NodeTy node_type;
  //! Edge iterator
  typedef typename boost::filter_iterator<is_edge, typename gNode::iterator> edge_iterator;
  //! Reference to edge data
  typedef typename gNode::EITy::reference edge_data_reference;
  //! Node iterator
  typedef boost::transform_iterator<makeGraphNode,
          boost::filter_iterator<is_node,
                   typename NodeListTy::iterator> > iterator;

  /**
   * Creates a new node holding the indicated data. Usually you should call
   * {@link addNode()} afterwards.
   */
  GraphNode createNode(const NodeTy& nd) {
    gNode* N = &(nodes.push(gNode(nd)));
    N->active = false;
    return GraphNode(N);
  }

  /**
   * Adds a node to the graph.
   */
  void addNode(const GraphNode& n, Galois::MethodFlag mflag = ALL) {
    GaloisRuntime::checkWrite(mflag | Galois::WRITE);
    GaloisRuntime::acquire(n, mflag);
    n->active = true;
  }

  //! Gets the node data for a node.
  NodeTy& getData(const GraphNode& n, Galois::MethodFlag mflag = ALL) const {
    assert(n);
    GaloisRuntime::checkWrite(mflag);
    GaloisRuntime::acquire(n, mflag);
    return n->data;
  }

  //! Checks if a node is in the graph
  bool containsNode(const GraphNode& n, Galois::MethodFlag mflag = ALL) const {
    assert(n);
    GaloisRuntime::acquire(n, mflag);
    return n->active;
  }

  /**
   * Removes a node from the graph along with all its outgoing/incoming edges
   * for undirected graphs or outgoing edges for directed graphs.
   */
  //FIXME: handle edge memory
  void removeNode(GraphNode n, Galois::MethodFlag mflag = ALL) {
    assert(n);
    GaloisRuntime::checkWrite(mflag | Galois::WRITE);
    GaloisRuntime::acquire(n, mflag);
    gNode* N = n;
    if (N->active) {
      N->active = false;
      if (!Directional && edges.mustDel())
	for (edge_iterator ii = edge_begin(n, NONE), ee = edge_end(n, NONE); ii != ee; ++ii)
	  edges.delEdge(ii->second());
      N->edges.clear();
    }
  }

  /** 
   * Adds an edge to graph, replacing existing value if edge already exists. 
   *
   * Ignore the edge data, let the caller use the returned iterator to set the
   * value if desired.  This frees us from dealing with the void edge data
   * problem in this API
   */
  edge_iterator addEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = ALL) {
    assert(src);
    assert(dst);
    GaloisRuntime::checkWrite(mflag | Galois::WRITE);
    GaloisRuntime::acquire(src, mflag);
    typename gNode::iterator ii = src->find(dst);
    if (ii == src->end()) {
      if (Directional) {
	ii = src->createEdge(dst, 0);
      } else {
	GaloisRuntime::acquire(dst, mflag);
	EdgeTy* e = edges.mkEdge();
	ii = dst->createEdge(src, e);
	ii = src->createEdge(dst, e);
      }
    }
    return boost::make_filter_iterator(is_edge(), ii, src->end());
  }

  edge_iterator addMultiEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = ALL) {
    assert(src);
    assert(dst);
    GaloisRuntime::checkWrite(mflag | Galois::WRITE);
    GaloisRuntime::acquire(src, mflag);
    typename gNode::iterator ii = src->end();
    if (ii == src->end()) {
      if (Directional) {
	ii = src->createEdge(dst, 0);
      } else {
	GaloisRuntime::acquire(dst, mflag);
	EdgeTy* e = edges.mkEdge();
	ii = dst->createEdge(src, e);
	ii = src->createEdge(dst, e);
      }
    }
    return boost::make_filter_iterator(is_edge(), ii, src->end());
  }

  //! Removes an edge from the graph
  void removeEdge(GraphNode src, edge_iterator dst, Galois::MethodFlag mflag = ALL) {
    assert(src);
    GaloisRuntime::checkWrite(mflag | Galois::WRITE);
    GaloisRuntime::acquire(src, mflag);
    if (Directional) {
      src->erase(dst.base());
    } else {
      GaloisRuntime::acquire(dst->first(), mflag);
      EdgeTy* e = dst->second();
      edges.delEdge(e);
      src->erase(dst.base());
      dst->first()->erase(src);
    }
  }

  //! Finds if an edge between src and dst exists
  edge_iterator findEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = ALL) {
    assert(src);
    assert(dst);
    GaloisRuntime::acquire(src, mflag);
    return boost::make_filter_iterator(is_edge(), src->find(dst), src->end());
  }

  /**
   * Returns the edge data associated with the edge. It is an error to
   * get the edge data for a non-existent edge.  It is an error to get
   * edge data for inactive edges. By default, the mflag is Galois::NONE
   * because edge_begin() dominates this call and should perform the
   * appropriate locking.
   */
  edge_data_reference getEdgeData(edge_iterator ii, Galois::MethodFlag mflag = NONE) const {
    assert(ii->first()->active);
    GaloisRuntime::checkWrite(mflag);
    GaloisRuntime::acquire(ii->first(), mflag);
    return *ii->second();
  }

  //! Returns the destination of an edge
  GraphNode getEdgeDst(edge_iterator ii) {
    assert(ii->first()->active);
    return GraphNode(ii->first());
  }

  //// General Things ////

  //! Returns an iterator to the neighbors of a node 
  edge_iterator edge_begin(GraphNode N, Galois::MethodFlag mflag = ALL) {
    assert(N);
    GaloisRuntime::acquire(N, mflag);

    if (GaloisRuntime::shouldLock(mflag)) {
      for (typename gNode::iterator ii = N->begin(), ee = N->end(); ii != ee; ++ii) {
	if (ii->first()->active)
	  GaloisRuntime::acquire(ii->first(), mflag);
      }
    }
    return boost::make_filter_iterator(is_edge(), N->begin(), N->end());
  }

  //! Returns the end of the neighbor iterator 
  edge_iterator edge_end(GraphNode N, Galois::MethodFlag mflag = ALL) {
    assert(N);
    // Not necessary; no valid use for an end pointer should ever require it
    //if (shouldLock(mflag))
    //  acquire(N);
    return boost::make_filter_iterator(is_edge(), N->end(), N->end());
  }

  /**
   * Returns an iterator to all the nodes in the graph. Not thread-safe.
   */
  iterator begin() {
    return boost::make_transform_iterator(
           boost::make_filter_iterator(is_node(),
				       nodes.begin(), nodes.end()),
	   makeGraphNode());
  }

  //! Returns the end of the node iterator. Not thread-safe.
  iterator end() {
    return boost::make_transform_iterator(
           boost::make_filter_iterator(is_node(),
				       nodes.end(), nodes.end()), 
	   makeGraphNode());
  }

  typedef iterator local_iterator;

  local_iterator local_begin() {
    return boost::make_transform_iterator(
           boost::make_filter_iterator(is_node(),
				       nodes.local_begin(), nodes.local_end()),
	   makeGraphNode());
  }

  local_iterator local_end() {
    return boost::make_transform_iterator(
           boost::make_filter_iterator(is_node(),
				       nodes.local_end(), nodes.local_end()), 
	   makeGraphNode());
  }

  /**
   * Returns the number of nodes in the graph. Not thread-safe.
   */
  unsigned int size() {
    return std::distance(begin(), end());
  }

  //! Returns the size of edge data.
  size_t sizeOfEdgeData() const {
    return gNode::EITy::sizeOfSecond();
  }

  FirstGraph() { }
};

}
}
#endif
