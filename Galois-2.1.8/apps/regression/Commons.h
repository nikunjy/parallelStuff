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
void normalize(std::vector<float, AlignmentAllocator<float,16> > &v) { 
	float sum = 0.0; 
	for (int i = 0; i < v.size(); i++) { 
		sum += v[i];
	}
	for (int i = 0; i < v.size(); i++) { 
		v[i] /= sum;
	}
}
void normalize(std::vector<float> &v) { 
	float sum = 0.0; 
	for (int i = 0; i < v.size(); i++) { 
		sum += v[i];
	}
	for (int i = 0; i < v.size(); i++) { 
		v[i] /= sum;
	}
}
#endif
