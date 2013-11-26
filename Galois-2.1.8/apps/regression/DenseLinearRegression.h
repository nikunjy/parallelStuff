#include "Commons.h"
using namespace std;
namespace DenseRegression {
struct Node { 
  std::vector <float,AlignmentAllocator<float,16> > featureValues;
  float outValue; 
};
typedef Galois::Graph::FirstGraph<Node,void,false> Graph;  //we should use LC graphs
typedef Graph::GraphNode GNode;
vector<float,AlignmentAllocator<float,16> > globalThetas;
Graph graph;
void readGraph(string fileName,int featureSize,int numSamples) { 
	fstream fin(fileName.c_str(),ios::in);
	vector<Node> equations;
	int dim,V; 
	fin>>dim>>V;
  dim = featureSize;
  V = numSamples;
  cout<<dim<<" "<<V;
  cout.flush();
  vector<float> means(dim+1,0);
  vector<pair<float,float> > variance(dim+1);
	for ( int i = 0; i < V; i++) { 
		Node node;
    string line;
    getline(fin,line);
    if (i == 0) { 
      continue;
    }
    istringstream iss(line);
    vector<string> tokens{istream_iterator<string>{iss},istream_iterator<string>{}};
    for (int k = 0; k <= dim ; k++) { 
			float value; 
      value = atof(tokens[k].c_str());
      means[k] += value;
      if (i == 1) { 
        variance[k].first = value;
        variance[k].second = value;
      } else { 
        variance[k].first = (variance[k].first > value) ? variance[k].first : value;
        variance[k].second = (variance[k].second < value) ? variance[k].second : value; 
      }
			if (k != dim) { 
        node.featureValues.push_back(value); 
      } else {
		    node.outValue = value;
      }
	  }
    equations.push_back(node);
  }
  for ( int k = 0; k <= dim ; k++) { 
    means[k]/=(double)V;
  }
	struct fillGraph { 
    vector<float> &means;
    vector<pair<float,float> > &var;
    fillGraph(vector<float> &m, vector<pair<float, float> >&v):
      means(m),var(v) {
    }
		void operator()(Node node) { 
			GNode gNode = graph.createNode(node);
      graph.addNode(gNode,Galois::NONE);
      auto &nd = graph.getData(gNode,Galois::NONE);
      vector<float,AlignmentAllocator<float,16> > &fv(nd.featureValues);
      for (int j = 0; j < fv.size(); j++) { 
        fv[j] = (fv[j]-means[j])/(var[j].first - var[j].second);
      }
      /*int dim = fv.size() + 1;
      nd.outValue = (nd.outValue - means[dim])/(var[dim].first - var[dim].second + 1);*/
		}
	};
	Galois::do_all(equations.begin(), equations.end(),fillGraph(means,variance),"make_graph");
}
typedef vector<float, AlignmentAllocator<float,16> > AlignedVector;
typedef GaloisRuntime::PerThreadStorage<vector<float,AlignmentAllocator<float,16> > > threadF; 
typedef GaloisRuntime::PerThreadStorage<float> threadG;	
struct GD { 
	threadF &localGains; 
	threadG &gainValue;
  int featureSize;
	GD(threadF &lg, threadG &gv,int featureSize):
    localGains(lg),gainValue(gv) {
     this->featureSize = featureSize; 
	}
	void operator()(GNode node) { 
		auto &nd = graph.getData(node,Galois::NONE);
		vector <float,AlignmentAllocator<float,16> > &localGain(*localGains.getLocal());
		float error = 0;
		float expectedValue = 0.0;
#ifdef USE_SSE
    int index = 0;
    __m128 a1,x1,r1,a2,x2,r2;
    __m128 sum = _mm_setzero_ps();
    __m128 sum1 = _mm_setzero_ps();
    for(size_t j=0;j<featureSize;j+=4,index+=4) {
        float *theta = &(globalThetas[index]);
        float *f = &(nd.featureValues[index]);
        a1 = _mm_load_ps(theta);
        x1 = _mm_load_ps(f);
        r1 = _mm_mul_ps(a1,x1);
        sum1 = _mm_add_ps(r1,sum1);
    }
     _mm_hadd_ps(sum1,sum1);
     _mm_hadd_ps(sum1,sum1);
     _mm_store_ss(&expectedValue,sum1);
#else
		for (int i = 0; i < featureSize; i++) {
			expectedValue += globalThetas[i] * nd.featureValues[i];
		}
#endif
		error = (expectedValue - nd.outValue);
		float &gain(*gainValue.getLocal());
		gain = error;
#ifdef USE_SSE
    index = 0; 
    a1 = _mm_load1_ps(&error);
    for (size_t j=0;j<featureSize;j+=4,index+=4) {
        float *f = &(nd.featureValues[index]);
        x1 = _mm_load_ps(f);
        r1 = _mm_mul_ps(a1,x1);
        float *result = &(localGain[index]);
        sum = _mm_load_ps(result);
        sum = _mm_add_ps(r1,sum);
        _mm_store_ps(result,sum);
    }
#else
		for (int i = 0; i < featureSize; i++) {
			localGain[i]+= error * nd.featureValues[i];
		}
#endif
	}
};
struct SGD {
  threadF &perThreadWeights; 
  float alpha;
  int batchSize;
  int featureSize; 
  SGD(threadF &ptw,float alpha, int fs, int batch = 10000):perThreadWeights(ptw) { 
    this->featureSize = fs;
    this->alpha = alpha;
    this->batchSize = batch;
  }
  void operator()(int num, Galois::UserContext<int>& ctx) {
    typedef std::vector<GNode, typename Galois::PerIterAllocTy::rebind<GNode>::other> TN;
    TN nodes(ctx.getPerIterAlloc());
    AlignedVector &localWeights(*perThreadWeights.getLocal());
    for (auto nb = graph.local_begin(), ne = graph.local_end(); nb != ne; nb++) {
      nodes.push_back(*nb);
    }
    srand(time(NULL));
    for(int count = 0; count < batchSize; count++) {
      int index = rand()%nodes.size();
      auto &nd = graph.getData(nodes[index],Galois::MethodFlag::NONE); 
      float expectedValue = 0.0;
      for (int i = 0; i < featureSize; i++) { 
        expectedValue += localWeights[i] * nd.featureValues[i];
      }
      float error = fabs(nd.outValue - expectedValue); 
      for (int i = 0; i < featureSize; i++) { 
        localWeights[i] -= alpha * error * nd.featureValues[i];
      }
      normalize(localWeights);
    }
  }
};
void do_gd (int featureSize) { 
  globalThetas.resize(featureSize);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1, 1);
  //Initialize globalThetas
  for ( int i = 0; i < featureSize; i++) { 
    globalThetas[i] = dis(gen);
  }
  float alpha = 0.005; //something 
  int iter = 0; 
  int total_iter = 200;
  Galois::StatTimer T;
  T.start();
  float prevError = 0.0;
  float deltaError = 0.0;
  do {
    threadF localGains; 
    threadG gainValues;
    for (int i = 0; i < localGains.size(); i++) { 
      localGains.getRemote(i)->resize(featureSize);
      float &gain(*gainValues.getRemote(i));
      gain = 0;
    }
    Galois::do_all_local(graph,GD(localGains,gainValues,featureSize));
    vector <float> globalGain(featureSize);
    for (int i = 0; i < localGains.size(); i++) { 
      vector<float,AlignmentAllocator<float,16> > &localGain(*localGains.getRemote(i));
      for (int j = 0; j < localGain.size(); j++) {
        globalGain[j] += localGain[i];
      }
    }
    float totalError = 0.0; 
    for (int i = 0; i < gainValues.size(); i++) { 
      totalError += *gainValues.getRemote(i);
    }
    if (iter>0) { 
      if (totalError > prevError) { 
        alpha /= 2;
        cout<<"New Alpha :"<<alpha<<endl; 
      } else if (iter % 50 == 0) { 
        alpha *= 2; 
      }
      deltaError = totalError - prevError;
      cout<<"Delta Error :" <<deltaError<<endl;
    }
    cout<<totalError<<endl;
    for (int i = 0; i < featureSize; i ++) { 
      globalThetas[i] -= alpha * globalGain[i];
      globalGain[i] = 0;
    }
    //Using gainValues do:
    //TODO specify termination condition here 
    //Change learning rate here if the convergence is too slow 
    //Change Learning rate here if the gain not monotonically decreasing
    prevError = totalError;
    iter++;  
    if ( iter > 1 && fabs(deltaError) - 0.001 <= 0) { 
      cout<<"Converged in number of Iterations : "<<iter;
      break;
    }
  }while(iter < total_iter);
  T.stop(); 
  cout<<"Time spent "<<T.get()<<" ms "<<endl;
}
struct calcError { 
  threadG &errors;
  int featureSize;
  calcError(threadG &e, int fs):errors(e) {
    this->featureSize = fs;
  }
  void operator()(int t) { 
    for (auto nb = graph.local_begin(),ne = graph.local_end(); nb!=ne; 
      nb++) { 
      auto &nd = graph.getData(*nb); 
      float expectedValue = 0.0;
      for (int i = 0; i < featureSize; i++) { 
        expectedValue += globalThetas[i] * nd.featureValues[i];
      }
      float &error(*errors.getLocal()); 
      error  += fabs(nd.outValue - expectedValue); 
    }
  }
};
void do_sgd (int featureSize) {
  Galois::StatTimer T;
  T.start();
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
      AlignedVector &threadWeights(*perThreadWeights.getRemote(i));
      threadWeights.resize(featureSize);
      for ( int j = 0; j < globalThetas.size(); j++) { 
        threadWeights[j] = globalThetas[j];
      }
    }
    vector<int> dummy(totalThreads);
    Galois::for_each(dummy.begin(),dummy.end(),SGD(perThreadWeights, alpha, featureSize, batchSize/totalThreads));
    AlignedVector tempThetas(globalThetas.size());
    copy(globalThetas.begin(),globalThetas.end(),tempThetas.begin());
    for (int i = 0 ; i < totalThreads; i++) { 
      AlignedVector &threadWeights(*perThreadWeights.getRemote(i));
      for ( int j = 0; j < threadWeights.size(); j++) { 
        globalThetas[j] += threadWeights[j];
      }
    }
    normalize(globalThetas);
    threadG perThreadError; 
    for (int i = 0; i < perThreadError.size(); i++) { 
      float &error(*perThreadError.getRemote(i)); 
      error = 0;
    }
    Galois::do_all(dummy.begin(),dummy.end(),calcError(perThreadError, featureSize));
    float totalError = 0; 
    for (int i = 0; i < totalThreads; i++) { 
      totalError += *perThreadError.getRemote(i);
    }
    if (iter_num > 0) {
      if (totalError > prevError) {
        copy(tempThetas.begin(),tempThetas.end(),globalThetas.begin());
        alpha /= 2;
        cout<<"New Alpha "<<alpha;
      } else { 
        prevError = totalError;
        cout<<"Total Error "<<totalError<<endl;
      }
    }
    if (iter_num == 0 )  {
      prevError = totalError;
    }
  }
  T.stop(); 
  cout<<"Time Taken "<<T.get()<<" ms "<<endl;
}
}