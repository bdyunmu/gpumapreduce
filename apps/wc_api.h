/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: Global.h 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)
	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#include "Panda.h"

#ifndef __USER_H__
#define __USER_H__


typedef struct
{
	char* file; 
} WC_KEY_T;

typedef __align__(16) struct
{
	int line_offset;
	int line_size;
} WC_VAL_T;

extern __device__ void panda_gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context pgc);

extern __device__ void panda_gpu_core_reduce(void *key, val_t* vals, int keySize, int valCount, panda_gpu_context pgc);

extern void panda_cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context *pcc, int map_task_idx);

extern __device__ void panda_gpu_core_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx);

extern __device__ int panda_gpu_core_compare(const void *key_a, int len_a, const void *key_b, int len_b);

extern void panda_gpu_card_map(void *key, void *val, int keySize, int valSize, panda_gpu_card_context *pgcc, int map_task_idx);

extern void panda_gpu_card_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_card_context *pgcc, int map_task_idx);

extern void panda_gpu_card_reduce(void *key, val_t* vals, int keySize, int valCount, panda_gpu_card_context* pgcc);

extern void panda_cpu_map(void *KEY, void*VAL, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx);

extern int panda_cpu_compare(const void *key_a, int len_a, const void *key_b, int len_b);

extern void cpu_map(void *KEY, void*VAL, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx);

extern void cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context* d_g_state);

extern int cpu_compare(const void *d_a, int len_a, const void *d_b, int len_b);

extern int gpu_card_compare(const void *d_a, int len_a, const void *d_b, int len_b);

extern void panda_cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context* pcc);

extern __device__ void panda_gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx);

extern __device__ void panda_gpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx);

extern __device__ void gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context pgc);

extern __device__ int gpu_compare(const void *d_a, int len_a, const void *d_b, int len_b);

#endif
