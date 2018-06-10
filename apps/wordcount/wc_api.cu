/*
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	Panda: a MapReduce Framework on GPUs and CPUs
	
	File: wc_api.cu 
	First Version:		2012-07-01 V0.1
	Last Updates:		2018-05-14
	Developer: Hui Li (huili@ruijie.com.cn)
*/

#include "Panda.h"
#include "PandaAPI.h"

namespace panda{

__device__ void panda_gpu_core_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx){
	//PandaGPUEmitCombinerOutput(KEY,&count,keySize,sizeof(int),pgc, map_task_idx);
}

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
				PandaEmitGPUMapOutput(word, one, ws, sizeof(int), pgc, map_task_idx);
			}//if
			valSize = valSize - ws;
			if(valSize<=0)
				break;
		}//while
		
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

}
