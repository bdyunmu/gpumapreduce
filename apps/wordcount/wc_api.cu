/*
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	Panda: a MapReduce Framework on GPUs and CPUs
	
	File: wc_api.cu 
	First Version:		2012-07-01 V0.10
	Last Updates:		2024-01-14 V0.62
	Developer: Hui Li (huili@ruijie.com.cn)
*/

#include "Panda.h"
#include "PandaAPI.h"
#include <stdio.h>
#include <string.h>

namespace panda{

__device__ unsigned int is_delim(char c, char *delim)
{
    while(*delim != '\0')
    {
        if(c == *delim)
            return 1;
        delim++;
    }
        return 0;
}

__device__ char *my_strtok(char *srcString, char *delim)
{
    static char *backup_string; // start of the next search
    if(!srcString)
    {
        srcString = backup_string;
    }
    if(!srcString)
    {
     // user is bad user
        return NULL;
    }
     // handle beginning of the string containing delims
    while(1)
    {
        if(is_delim(*srcString, delim))
        {
        srcString++;
        continue;
        }
        if(*srcString == '\0')
        {
        // we've reached the end of the string
        return NULL; 
        }
        break;
    }
    char *ret = srcString;
    while(1)
    {
        if(*srcString == '\0')
        {
        /*end of the input string and
         next exec will return NULL*/
         backup_string = srcString;
         return ret;
        }
        if(is_delim(*srcString, delim))
        {
         *srcString = '\0';
	 backup_string = srcString + 1;
	 return ret;
	}
	srcString++;
    }//while
}

__device__ size_t my_strlen(const char *str){
	const char *s = str;
	while(*s){
	s++;
	}
	return s - str;
}
__device__ void panda_gpu_core_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx){
	//PandaGPUEmitCombinerOutput(KEY,&count,keySize,sizeof(int),pgc, map_task_idx);
}//void

__device__ void panda_gpu_core_map(void *KEY, void *VAL, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx){

		int ws=0; //word size
		char *p = (char *)VAL;
		int *one = (int *)malloc(sizeof(int));
		*one = 1;
		char delimiters[] = " \n\t\"/,.;:?!-_()[]{}+=*&<>#@%0123456789";
		char *word = my_strtok(p,delimiters);
		while(word!=NULL){
			ws = my_strlen(word);
			printf("pgc word:%s len:%d\n",word,ws);
			PandaEmitGPUMapOutput(word, one, ws, sizeof(int), pgc, map_task_idx);
			word = my_strtok(NULL,delimiters);
		}
		
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
		char *word = strtok(p,delimiters);
		while(word!=NULL)
		{
			printf("pgc word:%s len:%d\n",word,strlen(word));
			ws = strlen(word);
			PandaEmitCPUMapOutput(word,one, ws, sizeof(int), pcc, map_task_idx);
			word = strtok(NULL,delimiters);
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
