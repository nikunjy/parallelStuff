#include<iostream>
#include<sstream>
#include<vector>
#include<stdio.h>
#include<ctime>
#include<omp.h>
#include<fstream>
#include<math.h>
#include<map>
#include<set>
#include<algorithm>
using namespace std;
struct Node { 
	int start; 
	int edges;
	Node() { 
		start = -1;
		edges = 0;
	}
	Node(int start,int edges) {
		this->start = start;
		this->edges = edges;
	}
};
bool comp(const pair<long,double> &p1,const pair<long,double> &p2) {
	return p1.second > p2.second;
}
double kandalltau(vector<double> &r1, vector<double> &r2) { 
	omp_set_num_threads(16); 
	int chunksize = r1.size()/16;
	vector<pair<long,double> > rp1(r1.size()),rp2(r1.size());	
	#pragma omp parallel for shared(rp1,rp2,r1,r2) schedule(static,chunksize)
	for(int i=0;i<r1.size();i++) { 
		rp1[i] = make_pair(i,r1[i]);
		rp2[i] = make_pair(i,r2[i]);
	}	
	sort(rp1.begin(),rp1.end(),comp);
	sort(rp2.begin(),rp2.end(),comp);
	set<long> setRank;
	for(int i=0;i<100;i++) {
		setRank.insert(rp1[i].first);
		setRank.insert(rp2[i].first);
	}
	vector<long> rankset;
	set<long>::iterator it = setRank.begin(); 
	while(it!=setRank.end()) {
		rankset.push_back(*it);
		it++;
	}
	double tau = 0;
	for (int i=0;i<rankset.size();i++) {
			int rankinFirst = -1; 
			int rankinSecond = -1; 
			for(int j=0;j<100;j++) {
				if(rp1[j].first == rankset[i]) 
					rankinFirst = j;
				if(rp2[j].first == rankset[i]) 
					rankinSecond = j;
			}

			if(rankinFirst == -1) {
				#pragma omp parallel for shared(rankset,i,rankinFirst,rp1,chunksize) schedule(static,chunksize)
				for(int k=0;k<rp1.size();k++) { 
						if(rp1[k].first == rankset[i]) {
							rankinFirst = k;
						}
				}
			}
			if(rankinSecond == -1) {
				#pragma omp parallel for shared(rankset,i,rankinSecond,rp2,chunksize) schedule(static,chunksize)
				for(int k=0;k<rp2.size();k++) { 
						if(rp2[k].first == rankset[i]) {
							rankinSecond = k;
						}
				}
			}
		for(int j=i+1;j<rankset.size();j++) {
			int rankJinFirst = -1; 
			int rankJinSecond = -1; 
			for(int k=0;k<100;k++) {
				if(rp1[k].first == rankset[j]) 
					rankJinFirst = k;
				if(rp2[k].first == rankset[j]) 
					rankJinSecond =k;
			}

			if(rankJinFirst == -1) {
				#pragma omp parallel for shared(rankset,j,rankJinFirst,rp1,chunksize) schedule(static,chunksize)
				for(int k=0;k<rp1.size();k++) { 
						if(rp1[k].first == rankset[j]) {
							rankJinFirst = k;
						}
				}
			}
			if(rankJinSecond == -1) {
				#pragma omp parallel for shared(rankset,j,rankJinSecond,rp2,chunksize) schedule(static,chunksize)
				for(int k=0;k<rp2.size();k++) { 
						if(rp2[k].first == rankset[j]) {
							rankJinSecond = k;
						}
				}
			}
			bool comp1 = (rankinFirst < rankinSecond);
			bool comp2 = (rankJinFirst < rankJinSecond);
			if(comp1!=comp2) {
				tau++;
			}
		}
	}

	double denominator = rankset.size() * (rankset.size()-1);
	denominator/=2.0;
	return tau/denominator;		
}
int main(int argc, char** argv) {
	int num_threads = atoi(argv[1]);
	char *filename = argv[2];
	cout<<filename;
	fstream fin(filename,ios::in); 
	#define cin fin
	omp_set_num_threads(num_threads);
	cout<<omp_get_max_threads()<<endl;
	double beta = 0.85;
	double epsilon = 0.1; 
	long V,E;
	E = 57708624;
	V = 23947346;
	vector<Node> nodes(V);
	vector<long> edges(E); 
	//read the graph
	int prev = 0;
	int count = 0;
	int start = 0;
	for(int j=0;j<E;j++) {	
		int src,dst; 
		cin>>src>>dst;
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
	double t1 = omp_get_wtime(); 
	vector<vector<double> > rank(2);
	rank[0].resize(V); 
	rank[1].resize(V);
	int chunksize = V/num_threads;
	#pragma omp parallel for default(none) shared(rank,V,chunksize,num_threads,std::cout)
	for(int t=0;t<num_threads;t++) { 
		int thread_start = t*chunksize; 
		int thread_end = (t+1)*chunksize; 
		if (t == num_threads -1) { 
			thread_end = V;
		}
		for (int i=thread_start;i<thread_end;i++) { 
			rank[0][i] = 1.0/(double)V;
		}
	}
	prev = 0; 
	int next = 1;
	double error = 0.0;
	int iter = 0;
	int lastSaved = prev;
	do{ 
	error = 0.0;
	#pragma omp parallel for default(none) shared(chunksize,prev,next,beta,V,rank,nodes,edges,num_threads,std::cout) reduction(+:error)
	for(int t=0;t<num_threads;t++) {
		int thread_start = t*chunksize; 
		int thread_end = (t+1)*chunksize; 
		if (t == num_threads -1) { 
			thread_end = V;
		}
		double diff = 0;
		for (int i = thread_start; i < thread_end; i++) { 
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
		
	}
	lastSaved = next;
	swap(prev,next);
	iter++;
	}while(error>0.0001);
	double t2 = omp_get_wtime(); 
	double time_spent  = double(t2 - t1);
	cout<<"Iterations "<<iter<<endl;
	cout<<time_spent<<endl;
	prev = (lastSaved ==1)?0:1;
	iter = 0;
	t1 = omp_get_wtime();
	#pragma omp parallel for default(none) shared(rank,V,chunksize,num_threads,prev)
	for(int t=0;t<num_threads;t++) { 
		int thread_start = t*chunksize; 
		int thread_end = (t+1)*chunksize; 
		if (t == num_threads -1) { 
			thread_end = V;
		}
		for (int i=thread_start;i<thread_end;i++) { 
			rank[prev][i] = 1.0/(double)V;
		}
	}
	do{ 
	error = 0.0;
	#pragma omp parallel for default(none) shared(chunksize,prev,beta,V,rank,nodes,edges,num_threads,std::cout) reduction(+:error)
	for(int t=0;t<num_threads;t++) {
		int thread_start = t*chunksize; 
		int thread_end = (t+1)*chunksize; 
		if (t == num_threads -1) { 
			thread_end = V;
		}
		double diff = 0;
		for (int i = thread_start; i < thread_end; i++) {
			double prevValue = rank[prev][i]; 
			rank[prev][i] = ((1.0 - beta)/(double)V); 
			double neigh_sum = 0.0; 
			int start = nodes[i].start; 
			int end = start + nodes[i].edges; 
			for (int j = start; j < end; j++) { 
				neigh_sum += rank[prev][edges[j]] / (double) nodes[edges[j]].edges;
			}
			rank[prev][i] += beta * neigh_sum; 
			error += fabs(rank[prev][i] - prevValue);
		}
	}
	iter++;
	}while(error > 0.0001);
	t2 = omp_get_wtime(); 
	cout<<"Iterations "<<iter<<endl;
	cout<<(t2-t1)<<endl;
	cout.flush();
	cout<<"Distance "<<kandalltau(rank[lastSaved],rank[prev]);
}
