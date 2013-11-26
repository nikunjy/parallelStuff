#include "Commons.h"
using namespace std;
namespace SparseRegression {

typedef map<int,float> iv;
struct Node { 
	std::vector<pair<int,float> >fvPair;
	int featureId;
};
vector<float> outValues;
typedef Galois::Graph::FirstGraph<Node,void,false> Graph;  //we should use LC graphs
typedef Graph::GraphNode GNode;
Graph graph;
void readGraph(string fileName,int &featureSize,int numSamples) {
  fstream fin(fileName.c_str(),ios::in);
  int V; 
  V = numSamples;
  map<int, int> counts;
  map<int, float> means;
  map<int, float> maxValues; 
  map<int, float> minValues;
  map<int, Node> idNodes;
  for ( int i = 0; i < V; i++) { 
    string line;
    getline(fin,line);
    istringstream iss(line);
    vector<string> tokens{istream_iterator<string>{iss},istream_iterator<string>{}};
    outValues.push_back(atof(tokens[0].c_str()));
    for (int j = 1; j < tokens.size(); j++) { 
    	int index = atoi(tokens[j].substr(0,tokens[j].find(":")).c_str());
    	tokens[j] = tokens[j].substr(tokens[j].find(":")+1);
    	float value = atof(tokens[j].c_str());
    	counts[index] += 1;
    	featureSize = max(featureSize, index);
    	means[index] += value;
    	maxValues[index] = max(maxValues[index],value);
    	minValues[index] = min(minValues[index],value);
    	if ( idNodes.find(i) == idNodes.end()) { 
    		Node node; 
    		node.featureId = index;
    		node.fvPair.push_back(make_pair(i,value));
    		idNodes[index] = node;
    	} else { 
    		idNodes[index].fvPair.push_back(make_pair(i,value));
    	}
    }
  }
  for (auto mean : means) {
  	means[mean.first] /= counts[mean.first];
  }
  struct fillGraph {
  	map<int, float> maxValues; 
  	map<int, float> minValues; 
  	map<int, float> means;
    fillGraph(iv maxv, iv minv, iv means){
    	this->maxValues = maxv; 
    	this->minValues = minv; 
    	this->means = means;
    }
    void operator()(pair<int,Node> idNode) {
      int index = idNode.first;
      GNode gNode = graph.createNode(idNode.second);
      graph.addNode(gNode,Galois::NONE);
      auto &nd = graph.getData(gNode,Galois::NONE);
      vector<pair<int,float> > &fv(nd.fvPair);
      for (auto p : fv) { 
      	p.second = (p.second - means[index]) / (maxValues[index] - minValues[index]);
      }
  	}	
  };
  Galois::do_all(idNodes.begin(), idNodes.end(),fillGraph(maxValues, minValues, means),"make_graph");
}
float loss_fn(float v1, float v2) { 
	return (v1 - v2);
}
typedef GaloisRuntime::PerThreadStorage<vector<float> > threadF; 
float lambda = 0.001;
struct SCD { 
	int numFeatures;
	int numSamples;
	int batchSize;
	threadF &perThreadWeights;
	SCD(int nf, int ns, int bs, threadF &ptw):perThreadWeights(ptw) { 
		this->numFeatures = nf;
		this->numSamples = ns;
		this->batchSize = bs;
	}
	void operator()(int t,Galois::UserContext<int>& ctx) { 
	  typedef std::vector<GNode, typename Galois::PerIterAllocTy::rebind<GNode>::other> TN;
	  typedef std::vector<float, typename Galois::PerIterAllocTy::rebind<float>::other> IP;
      TN nodes(ctx.getPerIterAlloc());
      IP innerProd(ctx.getPerIterAlloc());
      innerProd.resize(numSamples);
      for (auto nb = graph.local_begin(), ne = graph.local_end(); nb != ne; nb++) {
      	nodes.push_back(*nb);
      }
      srand(time(NULL));
      vector<float> &localWeights(*perThreadWeights.getLocal());
      for(int count = 0; count < batchSize; count++) {
      	int r = rand()%nodes.size();
      	auto &nd = graph.getData(nodes[r],Galois::NONE);
      	float sum = 0;
      	int featureId = nd.featureId;
      	for (auto ivpair : nd.fvPair) {
      		int eqId = ivpair.first; 
      		float value = ivpair.second;
      		sum += loss_fn(innerProd[eqId],outValues[eqId]) * value;
      	}
      	sum /= nd.fvPair.size();
      	float eta = 0.0;
      	if (localWeights[featureId] - sum > lambda) {
      		eta = -sum / 1 - lambda;
      	} else {                                                                                                            
      		if (localWeights[featureId] - sum < -lambda) {
      			eta = -sum + lambda;
      		} else {
      			eta = -localWeights[featureId]; 
      		}
    	}
    	localWeights[featureId] += eta;
    	for (auto ivpair : nd.fvPair) {
      		int eqId = ivpair.first; 
      		float value = ivpair.second;
      		innerProd[eqId] += eta * value;
      	}
      }
  	}
};
void do_scd(int featureSize, int numSamples) {
  Galois::StatTimer T;
  T.start();
  vector<float> globalThetas;
  globalThetas.resize(featureSize); 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1, 1);
  threadF perThreadWeights; 
  for ( int j = 0; j < globalThetas.size(); j++) { 
    globalThetas[j] = dis(gen);
  }
  normalize(globalThetas);
  float alpha = 0.05;
  float prevError = 0.0;
  int totalThreads = Galois::getActiveThreads();
  int batchSize = graph.size()/5; //Lets say we want to estimate using 20% of the data.
  for (int iter_num = 0; iter_num < 200; iter_num++) { 
    //Make all the weights (per thread) same.
    for ( int i = 0 ; i < perThreadWeights.size(); i++) {
      vector<float> &threadWeights(*perThreadWeights.getRemote(i));
      threadWeights.resize(featureSize);
      for ( int j = 0; j < globalThetas.size(); j++) { 
        threadWeights[j] = globalThetas[j];
      }
    }
    vector<int> dummy(totalThreads);
    Galois::for_each(dummy.begin(),dummy.end(),SCD(featureSize, numSamples, batchSize/totalThreads, perThreadWeights));
    vector<float> deltaThetas(globalThetas);
    vector<float> tempThetas(globalThetas.size());
    copy(globalThetas.begin(),globalThetas.end(),tempThetas.begin());
    for (int i = 0 ; i < totalThreads; i++) { 
      vector<float> &threadWeights(*perThreadWeights.getRemote(i));
      for ( int j = 0; j < threadWeights.size(); j++) { 
        deltaThetas[j]  += (globalThetas[j] - threadWeights[j]);
      }
    }
    for ( int j = 0; j < deltaThetas.size(); j++) { 
    	globalThetas[j] += deltaThetas[j];
    }
    normalize(globalThetas);
  }
  T.stop(); 
  cout<<"Time Taken "<<T.get()<<" ms "<<endl;
}
}
