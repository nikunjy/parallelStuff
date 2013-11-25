/** Linear Regression with multiple variables -*- C++ -*-
 * @file
 * Simple Linear regression with multivariables. With things like 
 	1. Mean Normalization 
 	2. Scaling
 	3. Learning rate adaptation
 * @section License
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
 * @author Nikunj Yadav < nikunj@cs.utexas.edu >
 */
#include "Commons.h"
#include <sstream>
#include <algorithm>
#include <iterator>
using namespace std;

const char* name = "Linear Regression with multivariables";
const char* desc = "Compute the gradient descent for linear regression with linear variables";
const char* url = "regression";
namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<int> fs(cll::Positional, cll::desc("featureSize"));
static cll::opt<int> ns(cll::Positional, cll::desc("Samples"));

vector<float,AlignmentAllocator<float,16> > globalThetas;
Graph graph;
void readGraph(string fileName,int &featureSize,int &numSamples) { 
	fstream fin(fileName.c_str(),ios::in);
	vector<Node> equations;
	int dim,V; 
	fin>>dim>>V;
  dim = fs;
  V = ns;
  cout<<dim<<" "<<V;
  cout.flush();
	featureSize = dim;
  numSamples = V;
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
      int dim = fv.size() + 1;
      nd.outValue = (nd.outValue - means[dim])/(var[dim].first - var[dim].second + 1);
		}
	};
	Galois::do_all(equations.begin(), equations.end(),fillGraph(means,variance),"make_graph");
}
#define USE_SSE
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
  int batchSize = 500;
  int featureSize; 
  SGD(threadF &ptw,float alpha):perThreadWeights(ptw) { 
    featureSize = fs;
    this->alpha = alpha;
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
        localWeights[i] -= (alpha * error) * nd.featureValues[i];
      }
    }
  }
};
void do_gd () { 
  globalThetas.resize(fs);
  int featureSize = fs;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1, 1);
  //Initialize globalThetas
  for ( int i = 0; i < featureSize; i++) { 
    globalThetas[i] = dis(gen);
  }
  float alpha = 0.05; //something 
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
      localGains.getRemote(i)->resize(fs);
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
      } else if (iter%50 == 0) { 
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
  calcError(threadG &e):errors(e) {
  }
  void operator()(int t) { 
    for (auto nb = graph.local_begin(),ne = graph.local_end(); nb!=ne; 
      nb++) { 
      auto &nd = graph.getData(*nb); 
      float expectedValue = 0.0;
      for (int i = 0; i < fs; i++) { 
        expectedValue += globalThetas[i] * nd.featureValues[i];
      }
      float &error(*errors.getLocal()); 
      error  += fabs(nd.outValue - expectedValue); 
    }
  }
};
void do_sgd () {
  globalThetas.resize(fs); 
  int featureSize = fs;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1, 1);
  threadF perThreadWeights; 
  for ( int j = 0; j < globalThetas.size(); j++) { 
    globalThetas[j] = dis(gen);
  }
  for ( int i = 0 ; i < perThreadWeights.size(); i++) {
    AlignedVector &threadWeights(*perThreadWeights.getRemote(i));
    threadWeights.resize(featureSize);
    for ( int j = 0; j < globalThetas.size(); j++) { 
      threadWeights[j] = globalThetas[j];
    }
  }
  float alpha = 0.05;
  float prevError = 0.0;
  for (int iter_num = 0; iter_num < 100; iter_num++) { 
    vector<int> dummy(perThreadWeights.size());
    Galois::for_each(dummy.begin(),dummy.end(),SGD(perThreadWeights, alpha));
    for (int i = 0; i< globalThetas.size(); i++)  {
      globalThetas[i] = 0.0; 
    }
    for (int i = 0 ; i < perThreadWeights.size(); i++) { 
      AlignedVector &threadWeights(*perThreadWeights.getRemote(i));
      for ( int j = 0; j < threadWeights.size(); j++) { 
        globalThetas[j] += threadWeights[j];
      }
    }
    for (int i = 0; i < perThreadWeights.size(); i++) { 
      globalThetas[i] /= perThreadWeights.size();
    }
    threadG perThreadError; 
    for (int i = 0; i < perThreadError.size(); i++) { 
      float &error(*perThreadError.getRemote(i)); 
      error = 0;
    }
    Galois::do_all(dummy.begin(),dummy.end(),calcError(perThreadError));
    float totalError = 0; 
    for (int i = 0; i < perThreadError.size(); i++) { 
      totalError += *perThreadError.getRemote(i);
    }
    if (iter_num > 0) {
      if (totalError > prevError) {
        alpha /= 2;
        cout<<"New Alpha "<<alpha<<endl;
      }
    }
    prevError = totalError;
    cout<<"Error "<<totalError<<endl;
  }
}

int main(int argc,char **argv) { 
	Galois::StatManager statManager;
	LonestarStart(argc, argv, name, desc, url);
  int featureSize,numSamples;
  //Galois::preAlloc(10000);
	readGraph(filename,featureSize,numSamples);
  cout<<"Graph read "<<featureSize<<" "<<numSamples<<endl;
  do_sgd(); 
  return 0;
}
