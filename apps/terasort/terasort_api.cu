/*

Copyright 2012 The Trustees of Indiana University.  All rights reserved.
Panda: a MapReduce Framework on GPUs and CPUs
File: main.cu 
Time: 2017-11-11 
Developer: Hui Li (huili@ruijie.com.cn)

*/


#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>

#include "PandaAPI.h"

__device__ void panda_gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context pgc){
}
__device__ int panda_gpu_core_compare(const void *key_a, int len_a, const void *key_b, int len_b){
return 0;
}
__device__ void panda_gpu_core_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx){
}
__device__ void panda_gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx){
}
__device__ void panda_gpu_core_reduce(void *key, val_t* vals, int keySize, int valCount, panda_gpu_context pgc){
}

int panda_cpu_compare(const void *key_a, int len_a, const void *key_b, int len_b)
{
return 0;
}

void panda_cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context *pcc, int map_task_idx){
                //PandaCPUEmitCombinerOutput(KEY,&count,keySize,sizeof(int),pcc, map_task_idx);
}//reduce2

void panda_cpu_map(void *KEY, void*VAL, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx){
}

void panda_cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context* pcc){

}
