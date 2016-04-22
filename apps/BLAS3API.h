/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.4
	File: Global.h 
	Time: 2013-05-25 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#include "Panda.h"

#ifndef __USER_H__
#define __USER_H__

#define MATRIX_BLOCK_SIZE 64






__device__ void gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx);

__device__ void gpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context *d_g_state, int map_task_idx);

__device__ void gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context d_g_state);

__device__ int gpu_compare(const void *d_a, int len_a, const void *d_b, int len_b);

int gpu_card_compare(const void *d_a, int len_a, const void *d_b, int len_b);

void cpu_map(void *KEY, void*VAL, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx);

void cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context* d_g_state);

int cpu_compare(const void *d_a, int len_a, const void *d_b, int len_b);

void cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context *d_g_state, int map_task_idx);



typedef struct
{

        float* matrix1;
        float* matrix2;
		float* matrix3;

		float* h_matrix1;
		float* h_matrix2;
		float* h_matrix3;

		//int test;

} MM_KEY_T;

typedef struct
{		
        int row;
        int col;
		
		int tbz;//thread block size
		int mbz;//matrix block size
		
        int row_dim;
        int col_dim;
} MM_VAL_T;


#endif
