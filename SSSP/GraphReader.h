#include "Graph.h"
#include <fstream>
using namespace std;
Graph& read_graph(string file) { 
	fstream fin(file.c_str(),ios::in); 
	int64_t V,E;
	fin>>V>>E; 
	Graph *graph = new Graph(V);
	int64_t src,dst; 
	for(int64_t i = 0; i < E; i++) { 
		fin>>src>>dst;
		graph->addEdge(src,dst);
	}
	return *graph;
}