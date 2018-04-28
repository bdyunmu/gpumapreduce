/*
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: map.cu 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2017-12-24
	Developer: Hui Li (lihui@indiana.edu)
	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
*/

//////////////
//WORD COUNT//
//////////////

#ifndef __USER_CU__
#define __USER_CU__

#include "Panda.h"
#include "wc_api.h"

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

       /* if (ka->i > kb->i)
                return 1;

        if (ka->i > kb->i)
                return -1;

        if (ka->i == kb->i)
                return 0;
	*/
}

__device__ void panda_gpu_core_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx){


}//reduce2

void panda_cpu_map(void *KEY, void*VAL, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx){
		
		int wsize = 0;
		char *start;
		char *p = (char *)VAL;

		while(1)
		{
			start = p;
			for(;*p>='A' && *p<='Z';p++);
			*p='\0';
			++p;
			wsize=(int)(p-start);
			if (wsize>6){
				char *wkey = (char *) malloc (wsize);
				memcpy(wkey,start,wsize);
				
				int *wc = (int *) malloc (sizeof(int));
				*wc=1;
				
				PandaEmitCPUMapOutput(wkey, wc, wsize, sizeof(int), pcc, map_task_idx);
			}//if
			valSize = valSize - wsize;
			if(valSize<=0)
				break;
		}//while
}//map2



void cpu_map(void *KEY, void*VAL, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){
		
}//map2

void panda_gpu_card_map(void *key, void *val, int keySize, int valSize, panda_gpu_card_context *pgcc, int map_task_idx){

}


__device__ void panda_gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx){

		
		int wsize = 0;
		char *start;
		char *p = (char *)VAL;
		int *wc = (int *) malloc (sizeof(int));
		*wc = 1;

		while(1)
		{
			start = p;
			for(; *p>='A' && *p<='Z'; p++);

			*p='\0';
			++p;
			wsize=(int)(p-start);
			if (wsize>6){
				char *wkey = start;
				PandaGPUEmitMapOutput(wkey, wc, wsize, sizeof(int), pgc, map_task_idx);
			}//if
			valSize = valSize - wsize;
			if(valSize<=0)
				break;
		}//while
		
		__syncthreads();
		
}//map2

__device__ void gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){
				//GPUEmitMapOutput(wkey, wc, wsize, sizeof(int), d_g_state, map_task_idx);
}//map2

void panda_cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context *pcc, int map_task_idx){
		
		//int *count = (int *) malloc (sizeof(int));
		int count = 0;
		for (int i=0;i<valCount;i++){
			 count += *((int *)(VAL[i].val));
		}//for

		PandaCPUEmitCombinerOutput(KEY,&count,keySize,sizeof(int),pcc, map_task_idx);
		
}//reduce2

void panda_gpu_card_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_card_context *pgcc, int map_task_idx){
}

__device__ void panda_gpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx){
		
		//int *count = (int *) malloc (sizeof(int));
		int count = 0;
		for (int i=0;i<valCount;i++){
			 count += *((int *)(VAL[i].val));
		}//

		PandaGPUEmitCombinerOutput(KEY,&count,keySize,sizeof(int),pgc, map_task_idx);
		
}//reduce2



/*__device__ void gpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context *d_g_state, int map_task_idx){
		//int *count = (int *) malloc (sizeof(int));
		int count = 0;
		for (int i=0;i<valCount;i++){
			 count += *((int *)(VAL[i].val));
		}//
		//GPUEmitCombinerOutput(KEY,&count,keySize,sizeof(int),d_g_state, map_task_idx);
}*///reduce2

void panda_gpu_card_reduce(void *key, val_t* vals, int keySize, int valCount, panda_gpu_card_context* pgcc){

}

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
        /*
        if (ka->i > kb->i)
                return 1;
        if (ka->i > kb->i)
                return -1;
        if (ka->i == kb->i)
                return 0;
        */

}

__device__ void panda_gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context pgc){

		int count = 0;
		for (int i=0;i<valCount;i++){
			count += *(int *)(VAL[i].val);
		}//
		
		//// deleted GPUEmitReduceOuput(KEY,&count,keySize,sizeof(int),&pgc);
		PandaGPUEmitReduceOutput(KEY,&count,keySize,sizeof(int),&pgc);
		
}//reduce2

/*
__device__ void gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context d_g_state){
		int count = 0;
		for (int i=0;i<valCount;i++){
			count += *(int *)(VAL[i].val);
		}//
		
		//GPUEmitReduceOuput(KEY,&count,keySize,sizeof(int),&d_g_state);
		
}*///reduce2


/*void cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context* d_g_state){
		int count = 0;
		for (int i=0;i<valCount;i++){
			count += *(int *)(VAL[i].val);
		}//
		CPUEmitReduceOutput(KEY,&count,keySize,sizeof(int),d_g_state);
		
}*///reduce2

void panda_cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context* pcc){

		int count = 0;
		for (int i=0;i<valCount;i++){
			count += *(int *)(VAL[i].val);
		}//
		
		PandaCPUEmitReduceOutput(KEY,&count,keySize,sizeof(int),pcc);
		
}//reduce2


/*int cpu_compare(const void *d_a, int len_a, const void *d_b, int len_b)
{
	char* word1 = (char*)d_a;
	char* word2 = (char*)d_b;
	for (; *word1 != '\0' && *word2 != '\0' && *word1 == *word2; word1++, word2++);
	if (*word1 > *word2) return 1;
	if (*word1 < *word2) return -1;
	return 0;
}*/

/*__device__ int gpu_compare(const void *d_a, int len_a, const void *d_b, int len_b)
{
	char* word1 = (char*)d_a;
	char* word2 = (char*)d_b;

	for (; *word1 != '\0' && *word2 != '\0' && *word1 == *word2; word1++, word2++);
	if (*word1 > *word2) return 1;
	if (*word1 < *word2) return -1;

	return 0;
}*/

#endif //__MAP_CU__
