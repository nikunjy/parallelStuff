#ifndef GRAPHREADER_H
#define GRAPHREADER_H
#include "Graph.h"
#include <fstream>
Graph& read_graph(std::string file) {
  return read_graph(file.c_str());
}
Graph& read_graph(char *file) { 
  std::fstream fin(file,std::ios::in); 
	int64_t V,E;
	fin>>V>>E; 
	Graph *graph = new Graph(V);
  graph->setNumEdges(E);
	int64_t src,dst,weight; 
	for(int64_t i = 0; i < E; i++) { 
		fin>>src>>dst>>weight;
    graph->addEdge(src,dst,weight);
	}
  fin.close();
	return *graph;
}
#endif
