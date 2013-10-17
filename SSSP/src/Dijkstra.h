#include "Graph.h"
#include <queue>
typedef std::pair<int64_t, int64_t> P;
struct compare { 
  bool operator()(const P& l, const P& r) { 
    return l.second > r.second;
  }
};
struct Dijkstra {
  void operator()(Graph &graph,int source,std::vector<int64_t>& relaxed_weights) {
    std::priority_queue<P,std::vector<P>,compare> Q;
    Q.push(std::make_pair(source,0));
    while (!Q.empty()) {
      P top = Q.top(); 
      Q.pop(); 
      int64_t src = top.first;
      int64_t weight = top.second; 
      if (relaxed_weights[src] < weight) { 
        continue;
      }
      relaxed_weights[src] = weight;
      std::vector<Edge> &edges(graph.getEdges(src));
      for (int i = 0; i < edges.size(); i++) { 
        Edge edge = edges[i];
        int64_t dst = edge.first; 
        int64_t edgeWeight = edge.second; 
        if (weight + edgeWeight < relaxed_weights[dst]) {
          relaxed_weights[dst] = weight + edgeWeight;
          Q.push(std::make_pair(dst,relaxed_weights[dst]));
        }
      }
    }
}

};

