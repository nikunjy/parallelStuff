#include <iostream>
#include <vector>
struct Node { 
	std::vector<int64_t> edges;
	void addEdge(int node) { 
		edges.push_back(node);
	}
};
class Graph { 
	private:
		std::vector<Node> nodes; 
	public:
	Graph(int64_t V) { 
		nodes.resize(V);
	}
	void addEdge(int source, int dest) { 
		nodes[source].addEdge(dest);
	}

};