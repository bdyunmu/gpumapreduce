/*

Copyright 2012 The Trustees of Indiana University.  All rights reserved.
CGL MapReduce Framework on GPUs and CPUs
Code Name: Panda 0.6
File: main.cu 
Time: 2017-11-11 
Developer: Hui Li (lihui@indiana.edu)
This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

*/


#include <mpi.h>
#include <panda/PreLoadedPandaChunk.h>
#include <panda/PandaMessage.h>
#include <panda/PandaMapReduceJob.h>
#include <panda/IntIntSorter.h>
#include <panda/PandaMapReduceWorker.h>
#include <cudacpp/Event.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <oscpp/Timer.h>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>

//-----------------------------------------------------------------------
//app name:
//C-means:  fuzzy data clustering algorithm 
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
		if (argc != 5)
		{
			ShowLog("Panda C-means");
			ShowLog("usage: %s [numPt] [cpu/gpu ratio] [numMapperPerCPU] [numMapperPerGPU]", argv[0]);
			exit(-1);//[Dimensions] [numClusters]
		}//if

		int numPt = atoi(argv[1]);
		int dim = 10;//atoi(argv[2]);
		int K = 10;//atoi(argv[3]);
		float ratio = atof(argv[2]);
		int numMapperCPU = atoi(argv[3]);
		int numMapperGPU = atoi(argv[4]);
	
		int maxIter = 10;//atoi(argv[5]);

		numMapperGPU = 1;
		numMapperCPU = 1;

		int num_gpu_core_groups = 0;
		int num_gpu_card_groups = 1;
		int num_cpus_groups = 0;
		float* h_points = GenPointsFloat(numPt, dim);
		float* h_cluster = GenInitCentersFloat(h_points, numPt, dim, K);
	
		int numgpus = 0;
		cudaGetDeviceCount(&numgpus);
		int global_dev_id = 0;
		
		panda::MapReduceJob  *job = new panda::PandaMapReduceJob(argc, argv, false,false,true);
		//panda::PandaMapReduceJob pjob(argc,argv,false,false,true);

		int rank, size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		
		job->setMessage(new panda::PandaFSMessage(true));
		//pjob.setMessage(new panda::PandaMPIMessage(true));

		float* d_points	 =	NULL;
		float* d_cluster =	NULL;
		int* d_clusterId =	NULL;
		float* d_tempClusters = NULL;
		float* d_tempDenominators = NULL;
		int dev_id = 0;
		cudaSetDevice(dev_id);

		cudaMalloc((void**)&d_points, numPt*dim*sizeof(int));
		cudaMemcpy(d_points, h_points, numPt*dim*sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_clusterId, numPt*sizeof(int));
		cudaMemset(d_clusterId, 0, numPt*sizeof(int));
		cudaMalloc((void**)&d_cluster, K*dim*sizeof(int));
		cudaMemcpy(d_cluster, h_cluster, K*dim*sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&d_tempClusters,K*dim*numMapperGPU*sizeof(float));
		cudaMemset(d_tempClusters, 0, sizeof(float)*K*dim*numMapperGPU);
		cudaMalloc((void**)&d_tempDenominators,numMapperGPU * K * sizeof(float));
		cudaMemset(d_tempDenominators, 0, sizeof(float)*K*numMapperGPU);
		cudaDeviceProp gpu_dev;
		cudaGetDeviceProperties(&gpu_dev, dev_id);
		ShowLog("Configure Device ID:%d: Device Name:%s", dev_id, gpu_dev.name);

#if 0		
		CMEANS_VAL_T val;
		//val.ptrPoints = (int *)d_points;
		//val.ptrClusters = (int *)d_cluster;
		val.d_Points = d_points;
		val.d_Clusters = d_cluster;
		//val.ptrChange = d_change;
		CMEANS_KEY_T key;
		key.dim = dim;
		key.K = K;
		//key.ptrClusterId = d_clusterId;
		int numPtPerGPU = numPt;
		int start = dev_id*numPtPerGPU;
		int end = start+numPtPerGPU;
		int numPtPerMap = (end-start)/numMapperGPU;
		ShowLog("GPU core numPtPerMap:%d startPt:%d  endPt:%d numPt:%d",numPtPerMap,start,end,numPt);

		int start_i,end_i;
		start_i = start;
		double t1 = PandaTimer();
		for (int j = 0; j < numMapperGPU; j++)
		{	
			end_i = start_i + numPtPerMap;
			if ( j < (end-start)%numMapperGPU)
				end_i++;
			ShowLog("start_i:%d, start_j:%d",start_i,end_i);
			//key.point_id = start_i;
			key.start = start_i;
			key.end = end_i;
			//key.global_map_id = dev_id*numMapperGPU+j;
			key.local_map_id = j;
			val.d_Points = d_points;
			val.d_tempClusters = d_tempClusters;
			val.d_tempDenominators = d_tempDenominators;
			job->addInput(new panda::VariousSizePandaChunk(&key,sizeof(CMEANS_KEY_T), &val,sizeof(CMEANS_VAL_T));
			//AddPandaTask(gpu_job_conf, &key, &val, sizeof(CMEANS_KEY_T), sizeof(CMEANS_VAL_T));
			job->execute();
			start_i = end_i;
		}//for

		double t2 = PandaTimer();
		ShowLog("Panda C-means take %f sec", t2-t1);
#endif

		return 0;
}//	

