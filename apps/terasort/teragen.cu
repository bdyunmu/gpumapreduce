/*

Copyright 2012 The Trustees of Indiana University.  All rights reserved.
Panda: a MapReduce Framework on GPUs and CPUs
File: main.cu 
Time: 2018-5- 9
Developer: Hui Li (huili@ruijie.com.cn)

*/

#include <mpi.h>
#include <panda/PreLoadedPandaChunk.h>
#include <panda/PandaMessage.h>
#include <panda/PandaMapReduceJob.h>
#include <panda/IntIntSorter.h>
#include <cudacpp/Event.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Stream.h>
#include <oscpp/Timer.h>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>
#include "PandaAPI.h"

#include "Unsigned16.h"
#include "Random16.h"


void generateRecord(byte *recBuf,Unsigned16 rand,Unsigned16 recordNumber){
	int i = 0;
	while(i < 10){
	recBuf[i] = rand.getByte(i);
	i += 1;
	}
	
	recBuf[10] = 0x00;
	recBuf[11] = 0x11;

	i = 0;
	while(i < 32){
	recBuf[12+i] = recordNumber.getHexDigit(i);
	i += 1;
	}

	recBuf[44] = 0x88;
	recBuf[45] = 0x99;
	recBuf[46] = 0xAA;
	recBuf[47] = 0xBB;

	i = 0;
	while(i < 12) {
		byte v = rand.getHexDigit(20+i);
		recBuf[48+i*4] = v;
		recBuf[49+i*4] = v;
		recBuf[50+i*4] = v;
		recBuf[51+i*4] = v;
		i += 1;
	}//

	recBuf[96] = 0xCC;
	recBuf[97] = 0xDD;
	recBuf[98] = 0xEE;
	recBuf[99] = 0xFF;
}

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
