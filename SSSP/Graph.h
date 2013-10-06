#ifndef GRAPH_H
#define GRAPH_H
#include <iostream>
#include <vector>
typedef std::pair<int64_t,int64_t> Edge;
struct Node { 
	std::vector<Edge> edges;
	void addEdge(int node,int weight) { 
		edges.push_back(std::make_pair(node,weight));
	}
  std::vector<Edge>& getEdges() { 
    return edges;
  }
};
class Graph { 
	private:
		std::vector<Node> nodes; 
    int64_t E;
	public:
	Graph(int64_t V) { 
		nodes.resize(V);
	}
  int64_t getNumNodes() { 
    return nodes.size(); 
  }
  int64_t getNumEdges() { 
    return E;
  }
  void setNumEdges(int64_t e) { 
    E = e;
  }
	void addEdge(int source, int dest, int weight) { 
		nodes[source].addEdge(dest,weight);
	}
  std::vector<Edge >& getEdges(int nodeId) {
    return nodes[nodeId].getEdges();
  }
  void printGraph() {
    for (int i = 0; i < nodes.size(); i++) { 
      std::cout<<"Node "<<i<<std::endl;
      std::cout<<"\t ";
      std::vector<Edge> &edges(nodes[i].getEdges());
      for (int j = 0 ; j < edges.size(); j++) { 
        std::cout<<"("<<edges[j].first<<" "<<edges[j].second<<") ";
      }
      std::cout<<std::endl;
    }
  }
};
#endif
