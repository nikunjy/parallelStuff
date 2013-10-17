#include "Graph.h"
#include <algorithm>
#include "GraphReader.h"
#include "Dijkstra.h"
#include "BellmanFord.h"
#include "Chaotic.h"
#include <ctime>
#include <map>
using namespace std;
const long INF = 2147483647;
namespace SSSP {
  bool verbose = false;
  int numThreads = 1;
}
map<string,int> algorithms;
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
void init() { 
  algorithms["Dijkstra"] = 0; 
  algorithms["BellmanFord"] = 1;
  algorithms["Chaotic"] = 2;
}
int main(int argc, char** argv) { 
  init();
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
  if (cmdOptionExists(argv, argv+argc, "-t")) {
    char* threads =  getCmdOption(argv, argv + argc, "-t"); 
    SSSP::numThreads = atoi(threads);
    cout<<"Operating at "<<SSSP::numThreads<<endl;
  }
  int algo = 0;
  if (cmdOptionExists(argv, argv+argc, "-algorithm")) {
    char* a_t =  getCmdOption(argv, argv + argc, "-algorithm"); 
    string algorithm = a_t;
    algo = algorithms[algorithm];
    cout<<algo<<endl;
  }
  char* filename = getCmdOption(argv, argv + argc, "-f");
  Graph g(read_graph(filename));
  if (SSSP::verbose) {
    cout<<"Graph Read"<<endl;
    cout<<g.getNumNodes()<<" "<<g.getNumEdges()<<endl;
  } 
  vector<int64_t> relaxed_weights(g.getNumNodes(),INF);
  double timeSpent = 0.0;
  if (algo == 0) {
      clock_t t1 = clock();
      cout<<"Using Dijkstra"<<endl;
      Dijkstra()(g,0,relaxed_weights);
      clock_t t2 = clock();
      timeSpent = (t2 - t1)/CLOCKS_PER_SEC;
  } else if (algo == 1) {
      cout<<"Using Bellman Ford";
      cout.flush();
     double t1 = omp_get_wtime(); 
      BellmanFord()(g,0,relaxed_weights);
     double t2 = omp_get_wtime();
      timeSpent = (t2 - t1);
  } else if (algo == 2) { 
     double t1 = omp_get_wtime(); 
      Chaotic()(g,0,relaxed_weights);
     double t2 = omp_get_wtime();
      timeSpent = (t2 - t1);
  }
  cout<<"Time Spent: "<<timeSpent<<endl;
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
