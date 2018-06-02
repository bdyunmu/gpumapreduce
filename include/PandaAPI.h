/*
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	Panda: a MapReduce Framework on GPUs and CPUs
	File: Global.h 
	Time: 2012-07-01 
	Developer: Hui Li (huili@ruijie.com.cn)
*/

#include "Panda.h"

#ifndef __USER_API_H__
#define __USER_API_H__

//extern __device__ void panda_gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context pgc);
//extern __device__ void panda_gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context pgc);
//extern __device__ void panda_gpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx);

extern __device__ void panda_gpu_core_reduce(void *key, val_t* vals, int keySize, int valCount, panda_gpu_context pgc);
extern __device__ void panda_gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx);
extern __device__ void panda_gpu_core_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx);

extern void panda_cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context *pcc, int map_task_idx);
extern void panda_cpu_map(void *KEY, void*VAL, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx);
extern void panda_cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context* pcc, int reduce_task_idx);

#endif
