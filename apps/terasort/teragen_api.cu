/*

Copyright 2012 The Trustees of Indiana University.  All rights reserved.
Panda: a MapReduce Framework on GPUs and CPUs
File: main.cu 
Time: 2018-5- 9
Developer: Hui Li (huili@ruijie.com.cn)

*/

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>
#include <dirent.h>

#include "PandaAPI.h"
#include "Unsigned16.h"
#include "Random16.h"
#include "TeraInputFormat.h"

__device__ void panda_gpu_core_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_gpu_context *pgc, int map_task_idx){
}
__device__ void panda_gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx){
}
__device__ void panda_gpu_core_reduce(void *key, val_t* vals, int keySize, int valCount, panda_gpu_context pgc){
}

void panda_cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context *pcc, int map_task_idx){
}//reduce2

void panda_cpu_map(void *KEY, void*VAL, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx){

	int rpp = TeraInputFormat::recordsPerPartition;

	Unsigned16 *one = new Unsigned16(1);		
	Unsigned16 *firstRecordNumber = new Unsigned16(*(int *)VAL*rpp);
	unsigned long h8 = firstRecordNumber->getHigh8();
	unsigned long l8 = firstRecordNumber->getLow8();
	printf("rpp:%d VAL:%d\n",rpp,*(int *)VAL);
	printf("firstRecordNumber h8:%ld l8:%ld\n",h8,l8);

	Unsigned16 *recordsToGenerate = new Unsigned16(rpp);

	Unsigned16 *recordNumber = new Unsigned16(*firstRecordNumber);

	Unsigned16 *lastRecordNumber = new Unsigned16(*firstRecordNumber);

	lastRecordNumber->add(*recordsToGenerate);

	Unsigned16 rand = Random16::skipAhead(*firstRecordNumber);
	h8 = rand.getHigh8();
	l8 = rand.getLow8();
	//printf("rand h8:%ld l8:%ld\n",h8,l8);

	byte* rowBytes = new byte[TeraInputFormat::RECORD_LEN];
	byte* key = new byte[TeraInputFormat::KEY_LEN];
	byte* value = new byte[TeraInputFormat::VALUE_LEN];
	memset(rowBytes,0,TeraInputFormat::RECORD_LEN);

	DIR *dp = NULL;
	char *path = TeraInputFormat::inputpath;
	if((dp = opendir(path))==NULL)
	{
	printf("not open %s\n",path);
	exit(-1);
	}else{
	closedir(dp);
	}
	char strfp[128];
	sprintf(strfp,"%s/INPUT%d",path,*(int *)VAL);
	FILE *fp = fopen(strfp,"wb");
	for(int i = 0;i<TeraInputFormat::recordsPerPartition;i++)
	{
	Random16::nextRand(rand);
	h8 = rand.getHigh8();
        l8 = rand.getLow8();
	//printf("next rand h8:%ld l8:%ld\n",h8,l8);
	TeraInputFormat::generateRecord(rowBytes,rand,*recordNumber);
	printf("key:::");
	for(int k = 0;k<10;k++){
		//rowBytes[k]+=100;
		printf("%3d",(unsigned char)rowBytes[k]);
	}
	printf(" \tid:::");
	for(int k = 0;k<40;k++){
		printf("%3d",(unsigned char)rowBytes[10+k]);
	}
	printf("\n");
	recordNumber->add(*one);
	TeraInputFormat::copyByte(rowBytes,key,0,TeraInputFormat::KEY_LEN);
	TeraInputFormat::copyByte(rowBytes,value,TeraInputFormat::KEY_LEN,TeraInputFormat::RECORD_LEN);
	fwrite(rowBytes,TeraInputFormat::RECORD_LEN,1,fp);	
	}
	fclose(fp);

}

void panda_cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, panda_cpu_context* pcc, int reduce_task_idx){
}
