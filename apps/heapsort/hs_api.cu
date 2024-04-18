/*
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	Panda: a MapReduce Framework on GPUs and CPUs cluster
	
	File: hs_api.cu 
	First Version:		2024-04-18 V0.42
	Last Updates:		2024-04-18 V0.42
	Developer: Hui Li (huili@ruijie.com.cn)
*/

#include "Panda.h"
#include "PandaAPI.h"
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <algorithm>

namespace panda{

void heapify(std::vector<int>&arr,int i,int n){
	int largest = i;
	int left = 2*i+1;
	int right = 2*i+2;
	if(left<n && arr[left] > arr[largest]){
		largest = left;
	}
	if(right<n && arr[right]>arr[largest]){
		largest = right;
	}
	if(largest!=i){
		std::swap(arr[i],arr[largest]);
		heapify(arr,largest,n);
	}
}
//建最大堆
void buildMaxHeap(std::vector<int>&arr, int n){
	for (int i=n/2-1;i>=0;i--){
	heapify(arr,i,n);	
	}
}


__device__ void panda_gpu_core_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx){
	//PandaGPUEmitCombinerOutput(KEY,&count,keySize,sizeof(int),pgc, map_task_idx);
}//void

__device__ void panda_gpu_core_map(void *KEY, void *VAL, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx){
		//PandaEmitGPUMapOutput(word, one, ws, sizeof(int), pgc, map_task_idx);
		__syncthreads();
}

__device__ void panda_gpu_core_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context pgc){
		int *count = (int *)malloc(sizeof(int));
		*count = 0;
		for (int i=0;i<valCount;i++){
			*count += *(int *)(VAL[i].val);
		}//
		PandaEmitGPUReduceOutput(KEY,count,keySize,sizeof(int),&pgc);
}

void panda_cpu_map(void *KEY, void*VAL, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx){

		int ws = 0;//word size
		char *p = (char *)VAL;
		int *one = (int *)malloc(sizeof(int));
		*one = 1;
		char delimiters[] = " \n\t\"/,.;:?!-_()[]{}+=*&<>#@%0123456789";
		char *word = NULL;//strtok(p,delimiters);
		while(word!=NULL)
		{
			printf("pgc word:%s len:%d\n",word,strlen(word));
			ws = strlen(word);
			PandaEmitCPUMapOutput(word,one, ws, sizeof(int), pcc, map_task_idx);
			word = NULL;//strtok(NULL,delimiters);
		}
}

void panda_cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context *pcc, int map_task_idx){
                int *count = (int *) malloc (sizeof(int));
                *count = 0;
                for (int i=0;i<valCount;i++){
                         *count += *((int *)(VAL[i].val));
                }//for
                PandaEmitCPUCombinerOutput(KEY,count,keySize,sizeof(int),pcc, map_task_idx);
}

void panda_cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context* pcc, int reduce_task_idx){
		int *count = new int[1];
		count[0] = 0;
		for (int i=0;i<valCount;i++){
			count[0] += *(int *)(VAL[i].val);
		}//
		PandaEmitCPUReduceOutput(KEY,(void *)count,keySize,sizeof(int),pcc,reduce_task_idx);
}

}//end of panda namespace
