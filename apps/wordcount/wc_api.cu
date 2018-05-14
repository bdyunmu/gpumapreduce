/*
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	Panda: a MapReduce Framework on GPUs and CPUs
	
	File: wc_api.cu 
	First Version:		2012-07-01 V0.1
	Last Updates:		2018-05-14
	Developer: Hui Li (huili@ruijie.com.cn)
*/

#ifndef __USER_CU__
#define __USER_CU__

#include "Panda.h"
#include "PandaAPI.h"

__device__ void panda_gpu_core_reduce(void *key, val_t* vals, int keySize, int valCount, panda_gpu_context pgc)
{
}

__device__ int panda_gpu_core_compare(const void *key_a, int len_a, const void *key_b, int len_b)
{
        int short_len = len_a>len_b? len_b:len_a;
	for(int i = 0;i<short_len;i++){
		if(((char *)key_a)[i]>((char *)key_b)[i])
		return 1;
		if(((char *)key_a)[i]<((char *)key_b)[i])
		return -1;
	}
	return 0;
}
__device__ void panda_gpu_core_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx){
}//reduce2

void panda_cpu_map(void *KEY, void*VAL, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx){
		
		int ws = 0;//word size
		char *start;
		char *p = (char *)VAL;

		while(1)
		{
			start = p;
			for(;*p>='A' && *p<='Z';p++);
			*p='\0';
			p++;
			ws=(int)(p-start);
			if (ws>6){
				char *word = (char *) malloc (ws);
				memcpy(word,start,ws);
				int *one = (int *)malloc(sizeof(int));
				*one=1;
				PandaEmitCPUMapOutput(word,one, ws, sizeof(int), pcc, map_task_idx);
			}//if
			valSize = valSize - ws;
			if(valSize<=0)
				break;
		}//while
}//map2

__device__ void panda_gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx){

		int ws=0; //word size
		char *start;
		char *p = (char *)VAL;
		int *one = (int *) malloc (sizeof(int));
		*one = 1;

		while(1)
		{
			start = p;
			for(; *p>='A' && *p<='Z'; p++);
			*p='\0';
			p++;
			ws=(int)(p-start);
			if (ws>6){
				char *word = start;
				PandaGPUEmitMapOutput(word, one, ws, sizeof(int), pgc, map_task_idx);
			}//if
			valSize = valSize - ws;
			if(valSize<=0)
				break;
		}//while
		
		__syncthreads();
		
}//map2

void panda_cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context *pcc, int map_task_idx){
		int *count = (int *) malloc (sizeof(int));
		*count = 0;
		for (int i=0;i<valCount;i++){
			 *count += *((int *)(VAL[i].val));
		}//for
		PandaCPUEmitCombinerOutput(KEY,count,keySize,sizeof(int),pcc, map_task_idx);
}//reduce2

#if 0
__device__ void panda_gpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx){
		//PandaGPUEmitCombinerOutput(KEY,&count,keySize,sizeof(int),pgc, map_task_idx);
}//reduce2
#endif

int panda_cpu_compare(const void *key_a, int len_a, const void *key_b, int len_b)
{
	int short_len = len_a>len_b? len_b:len_a;
	for(int i = 0;i<short_len;i++){
		if(((char *)key_a)[i]>((char *)key_b)[i])
		return 1;
		if(((char *)key_a)[i]<((char *)key_b)[i])
		return -1;
	}
	return 0;
}

__device__ void panda_gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context pgc){
		int *count = (int *)malloc(sizeof(int));
		*count = 0;
		for (int i=0;i<valCount;i++){
			*count += *(int *)(VAL[i].val);
		}//
		PandaGPUEmitReduceOutput(KEY,count,keySize,sizeof(int),&pgc);
}//reduce2


void panda_cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context* pcc){
		int *count = new int[1];
		count[0] = 0;
		for (int i=0;i<valCount;i++){
			count[0] += *(int *)(VAL[i].val);
		}//
		PandaCPUEmitReduceOutput(KEY,(void *)count,keySize,sizeof(int),pcc);
}//reduce2

#endif //__MAP_CU__
