#include "Commons.h"
#include "Graph.h"
#include "GraphReader.h"
#include "Dijkstra.h"
#include <iostream>
#include <algorithm>
using namespace std;
#define INF 987654321
namespace SSSP {
  bool verbose = false;
};
char* getCmdOption(char ** begin, char ** end, const std::string & option) {
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}
void printUsage() { 
  cout<<"sssp -f <filename> -o <output> -v"<<endl;
}
int main(int argc, char** argv) { 
  if (argc <= 1) {
    printUsage();
    return 0;
  }
  if (cmdOptionExists(argv, argv+argc, "-h")) {
    printUsage();
    return 0;
  }
  if (cmdOptionExists(argv, argv+argc, "-v")) {
    SSSP::verbose = true;
  }
  char* filename = getCmdOption(argv, argv + argc, "-f");
  Graph g(read_graph(filename));
  if (SSSP::verbose) {
    cout<<"Graph Read"<<endl;
  } 
  vector<int64_t> relaxed_weights(g.getNumNodes(),INF);
  Dijkstra()(g,0,relaxed_weights);
  if (cmdOptionExists(argv, argv+argc, "-o")) {
    char* outfile = getCmdOption(argv, argv + argc, "-o"); 
    fstream fout(outfile, ios::out);
    for (int i=0;i<relaxed_weights.size();i++) { 
      fout<<relaxed_weights[i]<<" ";
    } 
    fout.flush();
    fout.close();
  }
}
