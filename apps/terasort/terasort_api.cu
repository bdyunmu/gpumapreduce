/*
Copyright 2012 The Trustees of Indiana University.  All rights reserved.
Panda: a MapReduce Framework on GPUs and CPUs
File: main.cu 
Time: 2017-11-11 
Developer: Hui Li (huili@ruijie.com.cn)

*/


#include <cstdlib>
#include <cstdio>
#include <ctype.h>

#include "PandaAPI.h"
#include "TeraInputFormat.h"

typedef char byte;

__device__ void panda_gpu_core_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx){
}
__device__ void panda_gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx){
}
__device__ void panda_gpu_core_reduce(void *key, val_t* vals, int keySize, int valCount, panda_gpu_context pgc){
}

void panda_cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context *pcc, int map_task_idx){
                //PandaCPUEmitCombinerOutput(KEY,&count,keySize,sizeof(int),pcc, map_task_idx);
}//reduce2

void panda_cpu_map(void *KEY, void*VAL, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx){
	//byte *key = new byte[TeraInputFormat::KEY_LEN];
	//byte *value = new byte[TeraInputFormat::VALUE_LEN];
	//TeraInputFormat::copyByte((byte *)VAL,key,0,TeraInputFormat::KEY_LEN);
	//TeraInputFormat::copyByte((byte *)VAL,value,TeraInputFormat::KEY_LEN,TeraInputFormat::RECORD_LEN);
	PandaEmitCPUMapOutput(KEY,VAL,TeraInputFormat::KEY_LEN, TeraInputFormat::VALUE_LEN, pcc, map_task_idx);	
}

void panda_cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context* pcc, int reduce_task_idx){
	PandaEmitCPUReduceOutput(KEY, VAL, keySize, TeraInputFormat::VALUE_LEN, pcc, reduce_task_idx);
}
