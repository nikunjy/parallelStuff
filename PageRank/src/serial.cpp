#include<iostream>
#include<cstring>
#include<sstream>
#include<vector>
#include<stdio.h>
#include<ctime>
#include<fstream>
#include<math.h>
#include<algorithm>
using namespace std;
struct Node { 
	int start; 
	int edges;
	Node() { 
		start = 0;
		edges = 0;
	}
	Node(int start,int edges) {
		this->start = start;
		this->edges = edges;
	}
};
int main(int argc, char** argv) {
	int num_threads = atoi(argv[1]);
	char *filename = argv[2];
	cout<<filename;
	fstream fin(filename,ios::in); 
	double beta = 0.85;
	double epsilon = 1.0/10000.; 
	long V,E;
	E = 57708624;
	V = 23947346;
	vector<Node> nodes(V);
	vector<long> edges(E); 
	int count = 0;
	int start = 0;
	int prev = 0;
	for(int j=0;j<E;j++) {	
		int src,dst; 
		fin>>src>>dst;
		edges[j] = dst;
		if (src != prev) { 
			nodes[prev].start = start;
			nodes[prev].edges = count;
			start += count;
			count = 1;
			prev = src;
		}else {
			count++;
		}
	}
	nodes[prev].start = start; 
	nodes[prev].edges = count;
	cout<<"Graph Read "<<endl;
	clock_t t1 = clock(); 
	vector<vector<double> > rank(2);
	rank[0].resize(V); 
	rank[1].resize(V);
	int chunksize = V/num_threads;
	for (int i=0;i<V;i++) { 
		rank[0][i] = 1.0/(double)V;
	}
	prev = 0; 
	int next = 1;
	double error = 0.0;
	int iter = 0;	
	do{ 
	error = 0.0;
	for (int i = 0; i < V; i++) { 
		rank[next][i] = ((1.0 - beta)/(double)V); 
		double neigh_sum = 0.0; 
		int start = nodes[i].start; 
		int end = start + nodes[i].edges; 
		for (int j = start; j < end; j++) { 
			neigh_sum += rank[prev][edges[j]] / (double) nodes[edges[j]].edges;
		}
		rank[next][i] += beta * neigh_sum; 
		error += fabs(rank[next][i] - rank[prev][i]);
	}
	cout<<error<<endl;
	swap(prev,next);
	iter++;
	}while(error>epsilon);
	clock_t t2 = clock(); 
	double time_spent  = double(t2 - t1)/CLOCKS_PER_SEC;
	cout<<"Iterations "<<iter<<endl;
	cout<<time_spent;
	
	t1 = clock(); 
	prev = next = 0;
	for (int i=0;i<V;i++) { 
		rank[0][i] = 1.0/(double)V;
	}
	error = 0.0;
	iter = 0;	
	do{ 
	error = 0.0;
	for (int i = 0; i < V; i++) { 
		double prevValue = rank[prev][i];
		rank[prev][i] = ((1.0 - beta)/(double)V); 
		double neigh_sum = 0.0; 
		int start = nodes[i].start; 
		int end = start + nodes[i].edges; 
		for (int j = start; j < end; j++) { 
			neigh_sum += rank[prev][edges[j]] / (double) nodes[edges[j]].edges;
		}
		rank[next][i] += beta * neigh_sum; 
		error += fabs(rank[next][i] - prevValue);
	}
	iter++;
	}while(error>epsilon);
	t2=clock(); 
	cout<<"Time Spent "<<(t2-t1)/CLOCKS_PER_SEC;
	
}
