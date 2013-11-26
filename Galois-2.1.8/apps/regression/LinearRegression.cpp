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
#include "DenseLinearRegression.h"
#include "SparseLinearRegression.h"
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

int main(int argc,char **argv) { 
	Galois::StatManager statManager;
	LonestarStart(argc, argv, name, desc, url);
	//DenseRegression::readGraph(filename,fs,ns);
  //cout<<"Graph read "<<fs<<" "<<ns<<endl;
  //DenseRegression::do_sgd(fs); 
  int featureSize; 
  SparseRegression::readGraph(filename,featureSize,ns);
  cout<<"Graph Read " << featureSize<<" "<<ns;
  SparseRegression::do_scd(fs,ns);
  return 0;
}
