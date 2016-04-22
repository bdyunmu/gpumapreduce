/*

Copyright 2012 The Trustees of Indiana University.  All rights reserved.
CGL MapReduce Framework on GPUs and CPUs
Code Name: Panda 0.43
File: main.cu 
Time: 2013-06-11 
Developer: Hui Li (lihui@indiana.edu)

This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

*/


#include "mpi.h"
#if 0

//#include <panda/PreLoadedPandaChunk.h>
//#include <panda/PandaMPIMessage.h>
#include <panda/PandaMapReduceWorker.h>

#include "Global.h"

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>
#include "CmeansAPI.h"

//-----------------------------------------------------------------------
//usage: C-means datafile
//param: datafile 
//-----------------------------------------------------------------------

static float *GenPointsFloat(int numPt, int dim)
{
	float *matrix = (float*)malloc(sizeof(float)*numPt*dim);
	srand(time(0));
	for (int i = 0; i < numPt; i++)
		for (int j = 0; j < dim; j++)
			matrix[i*dim+j] = (float)((rand() % 100)/73.0);
	return matrix;
}//static float 

static float *GenInitCentersFloat(float* points, int numPt, int dim, int K)
{
	float* centers = (float*)malloc(sizeof(float)*K*dim);

	for (int i = 0; i < K; i++)
		for (int j = 0; j < dim; j++)
			centers[i*dim+j] = points[i*dim + j];
	return centers;
}//


int main(int argc, char** argv)
{
		if (argc != 4)
		{
			ShowLog("Panda C-means");
			ShowLog("usage: %s [numPt] [dim] [numK]", argv[0]);
			exit(-1);
		}//if

		int numPt	= atoi(argv[1]);
		int dim		= atoi(argv[2]);
		int K		= atoi(argv[3]);
		//float ratio = atof(argv[2]);
		int numMapperCPU = 1;//atoi(argv[3]);
		int numMapperGPU = 1;//atoi(argv[4]);
	
		int maxIter = 10;				//atoi(argv[5]);
		numMapperGPU = 1;
		numMapperCPU = 1;

		float* h_points		= GenPointsFloat(numPt, dim);
		float* h_cluster	= GenInitCentersFloat(h_points, numPt, dim, K);
	
		int numgpus = 0;
		cudaGetDeviceCount(&numgpus);
		//int global_dev_id = 0;
		panda::PandaMapReduceWorker  * job = new panda::PandaMapReduceWorker(argc, argv, true);
		//int rank, size;
		//MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		//MPI_Comm_size(MPI_COMM_WORLD, &size);
		job->setMessage (new panda::PandaFSMessage(true));
		job->setEnableCPU(false);
		job->setEnableGPU(false);
		job->setEnableGPUCard(true);
		
		/*
		float* d_points	 =	NULL;
		float* d_cluster =	NULL;
		int* d_clusterId =	NULL;
		float* d_tempClusters = NULL;
		float* d_tempDenominators = NULL;
		*/

		
		//checkCudaErrors(cudaSetDevice(dev_id));

		/*
		checkCudaErrors(cudaMalloc((void**)&d_points, numPt*dim*sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_points, h_points, numPt*dim*sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_clusterId, numPt*sizeof(int)));
		checkCudaErrors(cudaMemset(d_clusterId, 0, numPt*sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_cluster, K*dim*sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_cluster, h_cluster, K*dim*sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_tempClusters,K*dim*numMapperGPU*sizeof(float)));
		checkCudaErrors(cudaMemset(d_tempClusters, 0, sizeof(float)*K*dim*numMapperGPU));
		checkCudaErrors(cudaMalloc((void**)&d_tempDenominators,numMapperGPU * K * sizeof(float)));
		checkCudaErrors(cudaMemset(d_tempDenominators, 0, sizeof(float)*K*numMapperGPU));
		*/

		int dev_id = 0;
		CMEANS_VAL_T val;
		CMEANS_KEY_T key;
		
		int numPtPerGPU = numPt;
		int start		= dev_id*numPtPerGPU;
		int end			= start+numPtPerGPU;
		
		int numPtPerMap = (end-start)/numMapperGPU;
		ShowLog("GPU core numPtPerMap:%d startPt:%d  endPt:%d numPt:%d", numPtPerMap, start, end, numPt);

		int start_i,end_i;
		start_i		= start;
		double t1	= PandaTimer();

		for (int j = 0; j < numMapperGPU; j++)
		{	
			end_i = start_i + numPtPerMap;
			if ( j < (end-start)%numMapperGPU)
				end_i++;
			
			ShowLog("start_i:%d, start_j:%d  keySize:%d   valSize:%d", start_i, end_i,sizeof(CMEANS_KEY_T), sizeof(CMEANS_VAL_T));
			key.dim				= dim;
			key.K				= K;
			key.start			= start_i;
			key.end				= end_i;
			key.map_task_id		= j;

			/*
			val.d_Points			= d_points;
			val.d_Clusters			= d_cluster;
			val.d_tempClusters		= d_tempClusters;
			val.d_tempDenominators	= d_tempDenominators;
			*/
			job->addInput(new panda::VariousSizePandaChunk(&key,sizeof(CMEANS_KEY_T), &val,	sizeof(CMEANS_VAL_T)));
			start_i = end_i;

		}//for

		job->execute();
		delete job;
		double t2 = PandaTimer();
		ShowLog("Panda C-means take %f sec", t2-t1);
		DoLog2Disk("Panda C-means take %f sec numPt:%d  dim:%d K:%d", t2-t1, numPt, dim, K);
		return 0;
}//
#endif
