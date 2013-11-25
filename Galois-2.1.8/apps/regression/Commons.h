#ifndef COMMONS_H 
#define COMMONS_H
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
#include<smmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<immintrin.h>
#include "allocator.h"
struct Node { 
  std::vector <float,AlignmentAllocator<float,16> > featureValues;
	float outValue; 
};
typedef Galois::Graph::FirstGraph<Node,void,false> Graph;  //we should use LC graphs
typedef Graph::GraphNode GNode;
typedef vector<float, AlignmentAllocator<float,16> > AlignedVector;
typedef GaloisRuntime::PerThreadStorage<vector<float,AlignmentAllocator<float,16> > > threadF; 
typedef GaloisRuntime::PerThreadStorage<float> threadG;	
#endif
