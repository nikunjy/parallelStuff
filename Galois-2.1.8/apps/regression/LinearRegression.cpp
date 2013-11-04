/** Linear Regression with multiple variables -*- C++ -*-
 * @file
 *
 * Simple Linear regression with multivariables. With things like 
 	1. Mean Normalization 
 	2. Scaling
 	3. Learning rate adaptation
 *
 * @section License
 *
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
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#ifdef GALOIS_USE_EXP
#include "Galois/Runtime/ParallelWorkInline.h"
#endif
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include "boost/optional.hpp"

#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

const char* name = "Maximal Independent Set";
const char* desc = "Compute a maximal independent set (not maximum) of nodes in a graph";
const char* url = "independent_set";
namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);


/*
*Each Node represents an equation
*the form is y = a1x1 + a2x2 + a3x3+ ... 
* In the end we want to be able to find out the a1,a2,a3.... 
*/
struct Node { 
	vector <double> featureValues;
	double outValue;

};
vector<double> globalThetas;
typedef Galois::Graph::FirstGraph<Node,void,false> Graph;  //we should use LC graphs
typedef Graph::GraphNode GNode;
int featureSize;
Graph graph;
void readGraph(string fileName) { 
	fstream fin(fileName.c_str(),ios::in);
	vector<Node> equations;
	int dim,V; 
	fin>>dim>>V;
	featureSize = dim;
	for ( int i = 0; i < V; i++) { 
		Node node;
		for (int k = 0; k < dim ; k++) { 
			double value; 
			fin>>value;
			node.featureValues.push_back(value);
		}
		double value; 
		fin>>value; 
		node.outValue = value;
		equations.push_back(node);
	}
	struct fillGraph { 
		void operator()(Node node) { 
			GNode gNode = graph.createNode(node);
			graph.addNode(gNode,Galois::NONE);
		}
	};
	Galois::do_all(equations.begin(), equations.end(),fillGraph());
}
typedef GaloisRuntime::PerThreadStorage<vector<double> > threadF; 
typedef GaloisRuntime::PerThreadStorage<double> threadG;
struct Process { 
	threadF &localGains; 
	threadG &gainValue;
	Process(threadF &lg, threadG &gv):localGains(lg),gainValue(gv) { 
	}
	void operator()(GNode node) { 
		auto &nd = graph.getData(node,Galois::NONE);
		vector <double> &localGain(*localGains.getLocal());
		double error = 0;
		double expectedValue = 0.0;
		for (int i = 0; i < featureSize; i++) {
			expectedValue += globalThetas[i] * nd.featureValues[i];
		}
		error = (expectedValue - nd.outValue);
		double &gain(*gainValue.getLocal());
		gain = error;
		for (int i = 0; i < featureSize; i++) {
			localGain[i]+= error * nd.featureValues[i];
		}
	}
};
int main(int argc,char **argv) { 
	Galois::StatManager statManager;
	LonestarStart(argc, argv, name, desc, url);
	readGraph(filename);
	//TODO feature scaling and mean normalization for x values and y values 

	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);
	//Initialize globalThetas
	for ( int i = 0; i < featureSize; i++) { 
		globalThetas[i] = dis(gen);
	}
	double alpha = 0.1; //something 
	do {
		threadF localGains; 
		threadG gainValues;
		for (int i = 0; i < localGains.size(); i++) { 
			localGains.getRemote(i)->resize(featureSize);
			double &gain(*gainValues.getRemote(i));
			gain = 0;
		}
		Galois::do_all_local(graph,Process(localGains,gainValues));
		vector <double> globalGain(featureSize);
		for (int i = 0; i < localGains.size(); i++) { 
			vector<double> &localGain(*localGains.getRemote(i));
			for (int j = 0; j < localGain.size(); j++) {
				globalGain[j] += localGain[i];
			}
		}

		for (int i = 0; i < featureSize; i ++) { 
			globalThetas[i] -= alpha * globalGain[i];
		}

		//Using gainValues do:
		//TODO specify termination condition here 
		//Change learning rate hre if the convergence is too slow 
		//Change Learning rate here if the gain not monotonically decreasing

	}while(true);

}













