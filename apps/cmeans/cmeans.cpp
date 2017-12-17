/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: matrixutil.cpp 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */


#ifndef __MAP_CPP__
#define __MAP_CPP__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <CmeansAPI.h>


#if 0
typedef struct
{

	int local_map_id;
        int dim;
        int K;
        int* ptrClusterId;
        int start;
        int end;
        int global_map_id;

} CMEANS_KEY_T;

typedef struct
{
        int* ptrPoints;
        int* ptrClusters;

        float *d_tempClusters;
        float *d_tempDenominators;
        float *d_Clusters;
        float *d_Points;

} CMEANS_VAL_T;

typedef struct
{
   void * val;
   int valSize;
} val_t;
#endif

void cmeans_cpu_reduce_cpp(void *key, val_t* vals, int keySize, int valCount){

	CMEANS_KEY_T* pKey = (CMEANS_KEY_T*)key;
        int dim = pKey->dim;
        int K = pKey->K;

        float* myClusters = (float*) malloc(sizeof(float)*dim*K);
        float* myDenominators = (float*) malloc(sizeof(float)*K);
        //memset(myClusters,0,sizeof(float)*dim*K);
        //memset(myDenominators,0,sizeof(float)*K);

        float *tempClusters = NULL;
        float *tempDenominators = NULL;
		
        for (int i = 0; i < valCount; i++)
        {
                int index = pKey->local_map_id;

				CMEANS_VAL_T* pVal = (CMEANS_VAL_T*)(vals[i].val);
                tempClusters = pVal->d_tempClusters + index*K*dim;
                tempDenominators = pVal->d_tempDenominators+ index*K;
                for (int k = 0; k< K; k++){
                        for (int j = 0; j< dim; j++)
                                myClusters[k*dim+j] += tempClusters[k*dim+j];
                        myDenominators[k] += tempDenominators[k];
                }//for
        }//end for

        for (int k = 0; k< K; k++){
			for (int i = 0; i < dim; i++){
						//printf("K:%d dim:%d myDenominators[i]:%f",K,dim,myDenominators[i]);
                        myClusters[i] /= ((float)myDenominators[i]+0.0001);
						//printf("%f ",myClusters[i]);
			}//for
			//printf("\n");
        }//for
		
	free(myClusters);
	free(myDenominators);

}

void cmeans_cpu_map_cpp(void *key, void *val, int keySize, int valSize){
			
    CMEANS_KEY_T* pKey = (CMEANS_KEY_T*)key;
	CMEANS_VAL_T* pVal = (CMEANS_VAL_T*)val;
	
	int dim = pKey->dim;
	int K = pKey->K;
	int start = pKey->start;
	int end = pKey->end;
	int index = pKey->local_map_id;
	
	float *point	= (float*)(pVal->d_Points);
	float *cluster	= (float*)(pVal->d_Clusters);

	float * tempClusters = pVal->d_tempClusters+index*dim*K;
	float * tempDenominators = pVal->d_tempDenominators+index*K;

	float denominator = 0.0f;
	float membershipValue = 0.0f;

	float *distances = (float *)malloc(sizeof(float)*K);
	float *numerator = (float *)malloc(sizeof(float)*K);
	
	for(int i=0; i<K; i++){
		distances[i]=0.0f;
		numerator[i]=0.0f;
	}//for

	//printf("map_task_id 0:%d thread_id:%d\n",map_task_idx,THREAD_ID);
	for (int i=start; i<end; i++){
		float *curPoint = (float*)(pVal->d_Points + i*dim);
		for (int k = 0; k < K; ++k)
		{
			float* curCluster = (float*)(pVal->d_Clusters + k*dim);
			distances[k] = 0.0;
			//printf("dim:%d\n",dim);
			float delta = 0.0;	
			
			for (int j = 0; j < dim; ++j)
			{
				delta = curPoint[j]-curCluster[j];
				distances[k] += (delta*delta);
			}//for
			
			numerator[k] = powf(distances[k],2.0f/(2.0-1.0))+1e-30;
			denominator  = denominator + 1.0f/(numerator[k]+1e-30);
		}//for

		for (int k = 0; k < K; ++k)
		{
			membershipValue = 1.0f/powf(numerator[k]*denominator,(float)2.0);
			for(int d =0; d<dim; d++){
				tempClusters[k*dim+d] += (curPoint[d])*membershipValue;
			}
			tempDenominators[k]+= membershipValue;
		}//for 
	}//for
	//printf("map_task_id 1:%d\n",map_task_idx);
	free(distances);
	free(numerator);
	
	//TODO
	pKey->local_map_id = 0;
	pKey->end = 0;
	pKey->start = 0;
	pKey->global_map_id = 0;
}

#endif //__MAP_CPP__
