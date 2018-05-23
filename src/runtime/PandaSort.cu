/*

        Copyright 2012 The Trustees of Indiana University.  All rights reserved.
        Panda: co-processing SPMD computations on GPUs and CPUs.

        File: PandaSort.cu
        First Version:          2012-07-01 V0.1
        Last UPdates:           2018-04-28 v0.41
        Developer: Hui Li (huili@ruijie.com.cn)

*/

#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <cuda_runtime.h>

#ifndef _PANDA_SORT_CU_
#define _PANDA_SORT_CU_

#include "Panda.h"

void ExecutePandaSortBucket(panda_node_context *pnc)
{

	  int numRecvedBuckets = pnc->recv_buckets.counts.size();
	  ShowLog("numRecvedBuckets:%d",numRecvedBuckets);

	  keyvals_t *sorted_intermediate_keyvals_arr = NULL; 
          pnc->sorted_key_vals.sorted_intermediate_keyvals_arr = NULL;
          pnc->sorted_key_vals.sorted_keyvals_arr_len = 0;

	  char *key_0, *key_1;
	  int keySize_0, keySize_1;
	  char *val_0;
	  // char *val_1;
	  int valSize_0;
	  //int valSize_1;
	  for(int i=0; i<numRecvedBuckets; i++){
		char *keyBuff = pnc->recv_buckets.savedKeysBuff[i];
		char *valBuff = pnc->recv_buckets.savedValsBuff[i];
		int *counts = pnc->recv_buckets.counts[i];

		int *keyPosArray  = pnc->recv_buckets.keyPos[i];
		int *keySizeArray = pnc->recv_buckets.keySize[i];
		int *valPosArray  = pnc->recv_buckets.valPos[i];
		int *valSizeArray = pnc->recv_buckets.valSize[i];

		int maxlen	= counts[0];
		int keyBuffSize	= counts[1];
		int valBuffSize	= counts[2];

		for (int j=0; j<maxlen; j++){
			
			if( keyPosArray[j] + keySizeArray[j] > keyBuffSize ) 
				ErrorLog("(keyPosArray[j]:%d + keySizeArray[j]:%d > keyBuffSize:%d)", keyPosArray[j], keySizeArray[j] , keyBuffSize);

			key_0		= keyBuff + keyPosArray[j];
			keySize_0	= keySizeArray[j];

			int k = 0;
			for ( ;k<pnc->sorted_key_vals.sorted_keyvals_arr_len;k++){
				key_1		= (char *)(sorted_intermediate_keyvals_arr[k].key);
				keySize_1	= sorted_intermediate_keyvals_arr[k].keySize;
				int compare_res = cpu_compare(key_0,keySize_0,key_1,keySize_1);
				if(compare_res>0){
				  	continue;
				}
				else if(compare_res==0){		
				  	val_t *vals = sorted_intermediate_keyvals_arr[k].vals;
				  	int index   = sorted_intermediate_keyvals_arr[k].val_arr_len;
					sorted_intermediate_keyvals_arr[k].val_arr_len++;
					sorted_intermediate_keyvals_arr[k].vals = (val_t*)realloc(vals, 
						sizeof(val_t)*(sorted_intermediate_keyvals_arr[k].val_arr_len));
					val_0   = valBuff + valPosArray[j];
					valSize_0 = valSizeArray[j];
					sorted_intermediate_keyvals_arr[k].vals[index].val = (char *)malloc(sizeof(char)*valSize_0);
					sorted_intermediate_keyvals_arr[k].vals[index].valSize = valSize_0;
					memcpy(sorted_intermediate_keyvals_arr[k].vals[index].val, val_0, valSize_0);
					break;
				}else{
					int index = pnc->sorted_key_vals.sorted_keyvals_arr_len;
					pnc->sorted_key_vals.sorted_keyvals_arr_len++;
					sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, 
						sizeof(keyvals_t)*(pnc->sorted_key_vals.sorted_keyvals_arr_len));
					while(index>k){
						keyvals_t* kvalsp0 = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);
						keyvals_t* kvalsp1 = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index-1]);
						kvalsp0->keySize = kvalsp1->keySize;
						kvalsp0->key = kvalsp1->key;
						kvalsp0->vals = kvalsp1->vals;
						kvalsp0->val_arr_len = kvalsp1->val_arr_len;
						index--;
					}//while
					keyvals_t* kvalsp2 = (keyvals_t *)&(sorted_intermediate_keyvals_arr[k]);
					kvalsp2->keySize = keySize_0;
					kvalsp2->key = malloc(sizeof(char)*keySize_0);
					memcpy(kvalsp2->key,key_0,keySize_0);
					kvalsp2->vals = (val_t *)malloc(sizeof(val_t)*1);
					kvalsp2->val_arr_len = 1;
					val_0 = valBuff+valPosArray[j];
					valSize_0 = valSizeArray[j];
					kvalsp2->vals[0].valSize = valSize_0;
					kvalsp2->vals[0].val = (char *)malloc(sizeof(char)*valSize_0);
					memcpy(kvalsp2->vals[0].val, val_0, valSize_0);
					break;
				}//else
			}//for k
			if (k == pnc->sorted_key_vals.sorted_keyvals_arr_len){

			if (pnc->sorted_key_vals.sorted_keyvals_arr_len == 0) sorted_intermediate_keyvals_arr = NULL;

			int index = pnc->sorted_key_vals.sorted_keyvals_arr_len;
			pnc->sorted_key_vals.sorted_keyvals_arr_len++;
			sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, 
				sizeof(keyvals_t)*(pnc->sorted_key_vals.sorted_keyvals_arr_len));
			keyvals_t* kvalsp = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);

			kvalsp->keySize = keySize_0;
			kvalsp->key = malloc(sizeof(char)*keySize_0);
			memcpy(kvalsp->key, key_0, keySize_0);

			kvalsp->vals = (val_t *)malloc(sizeof(val_t)*1);
			kvalsp->val_arr_len = 1;

			if (valPosArray[j] + valSizeArray[j] > valBuffSize)
				ErrorLog("(valPosArray[j] + valSizeArray[j] > valBuffSize)");

			val_0   = valBuff + valPosArray[j];
			valSize_0 = valSizeArray[j];

			kvalsp->vals[0].valSize = valSize_0;
			kvalsp->vals[0].val = (char *)malloc(sizeof(char)*valSize_0);
			memcpy(kvalsp->vals[0].val, val_0, valSize_0);

			}//k
		}//j
	  }//i
	  pnc->sorted_key_vals.sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;
}//			

void ExecutePandaGPUSort(panda_gpu_context* pgc){

	cudaThreadSynchronize();

	int *count_arr = (int *)malloc(sizeof(int) * pgc->input_key_vals.num_input_record);
	cudaMemcpy(count_arr, pgc->intermediate_key_vals.d_intermediate_keyval_total_count, 
		sizeof(int)*pgc->input_key_vals.num_input_record, cudaMemcpyDeviceToHost);

	int total_count = 0;
	for(int i=0;i<pgc->input_key_vals.num_input_record;i++){
		total_count += count_arr[i];
	}//for
	//free(count_arr);

	ShowLog("GPU Total Count of Intermediate Records:%d num_input_record:%d",total_count,pgc->input_key_vals.num_input_record);
	cudaMalloc((void **)&(pgc->intermediate_key_vals.d_intermediate_keyval_arr),sizeof(keyval_t)*total_count);

	//int num_mappers = 1;
	//int num_blocks = (num_mappers + (NUM_THREADS)-1)/(NUM_THREADS);
	//int num_blocks = (pgc->input_key_vals->num_mappers + (NUM_THREADS)-1)/(NUM_THREADS);
	int numGPUCores = pgc->num_gpus_cores;//getGPUCoresNum();
	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
	dim3 grids(numBlocks, 1);


	cudaThreadSynchronize();
	//copyDataFromDevice2Host1<<<grids,blocks>>>(pgc);
	//note: memory overflow error here. 2018/4/30
	copyDataFromDevice2Host1<<<1,16>>>(*pgc);
	cudaThreadSynchronize();

	//TODO intermediate keyval_arr use pos_arr
	keyval_t * h_keyval_arr = (keyval_t *)malloc(sizeof(keyval_t)*total_count);
	cudaMemcpy(h_keyval_arr, pgc->intermediate_key_vals.d_intermediate_keyval_arr, 
		sizeof(keyval_t)*total_count, cudaMemcpyDeviceToHost);

pgc->intermediate_key_vals.h_intermediate_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	keyval_pos_t *h_intermediate_keyvals_pos_arr = pgc->intermediate_key_vals.h_intermediate_keyval_pos_arr;

	int totalKeySize = 0;
	int totalValSize = 0;

	for (int i=0;i<total_count;i++){

		//ErrorLog("[i:%d] totalValSize:%d totalKeySize:%d\n",i,totalValSize,totalKeySize);
		h_intermediate_keyvals_pos_arr[i].valPos= totalValSize;
		h_intermediate_keyvals_pos_arr[i].keyPos = totalKeySize;

		h_intermediate_keyvals_pos_arr[i].keySize = h_keyval_arr[i].keySize;
		h_intermediate_keyvals_pos_arr[i].valSize = h_keyval_arr[i].valSize;

		totalKeySize += (h_keyval_arr[i].keySize+3)/4*4;
		totalValSize += (h_keyval_arr[i].valSize+3)/4*4;
	}//for

	if ((totalValSize<=0)||(totalKeySize<=0)){
		ErrorLog("(totalValSize==0)||(totalKeySize==0) Warning!");
		pgc->sorted_key_vals.totalValSize = totalValSize;
		pgc->sorted_key_vals.totalKeySize = totalKeySize;
		pgc->sorted_key_vals.d_sorted_keyvals_arr_len = 0;
		return;	
		//exit(0);
	}//if

	pgc->sorted_key_vals.totalValSize = totalValSize;
	pgc->sorted_key_vals.totalKeySize = totalKeySize;

	ShowLog("cudaMalloc totalKeySize:%lf KB totalValSize:%lf KB, number of intermediate records:%d", 
				(double)(totalKeySize)/1024.0, (double)totalValSize/1024.0, total_count);
	cudaMalloc((void **)&pgc->intermediate_key_vals.d_intermediate_keys_shared_buff,totalKeySize);
	cudaMalloc((void **)&pgc->intermediate_key_vals.d_intermediate_vals_shared_buff,totalValSize);
	cudaMalloc((void **)&pgc->intermediate_key_vals.d_intermediate_keyval_pos_arr,sizeof(keyval_pos_t)*total_count);
	cudaMemcpy(pgc->intermediate_key_vals.d_intermediate_keyval_pos_arr, h_intermediate_keyvals_pos_arr, sizeof(keyval_pos_t)*total_count, cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
	///copyDataFromDevice2Host2<<<grids,blocks>>>(*pgc);
	copyDataFromDevice2Host2<<<1,16>>>(*pgc);
	cudaThreadSynchronize();

	pgc->intermediate_key_vals.h_intermediate_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	pgc->intermediate_key_vals.h_intermediate_vals_shared_buff = malloc(sizeof(char)*totalValSize);

	cudaMemcpy(pgc->intermediate_key_vals.h_intermediate_keys_shared_buff,pgc->intermediate_key_vals.d_intermediate_keys_shared_buff,sizeof(char)*totalKeySize,cudaMemcpyDeviceToHost);
	cudaMemcpy(pgc->intermediate_key_vals.h_intermediate_vals_shared_buff,pgc->intermediate_key_vals.d_intermediate_vals_shared_buff,sizeof(char)*totalValSize,cudaMemcpyDeviceToHost);

	//////////////////////////////////////////////
	cudaMalloc((void **)&pgc->sorted_key_vals.d_sorted_keys_shared_buff,totalKeySize);
	cudaMalloc((void **)&pgc->sorted_key_vals.d_sorted_vals_shared_buff,totalValSize);
	cudaMalloc((void **)&pgc->sorted_key_vals.d_keyval_pos_arr,sizeof(keyval_pos_t)*total_count);

	pgc->sorted_key_vals.h_sorted_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	pgc->sorted_key_vals.h_sorted_vals_shared_buff = malloc(sizeof(char)*totalValSize);

	char *sorted_keys_shared_buff = (char *)pgc->sorted_key_vals.h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff = (char *)pgc->sorted_key_vals.h_sorted_vals_shared_buff;

	char *intermediate_key_shared_buff = (char *)pgc->intermediate_key_vals.h_intermediate_keys_shared_buff;
	char *intermediate_val_shared_buff = (char *)pgc->intermediate_key_vals.h_intermediate_vals_shared_buff;
	memcpy(sorted_keys_shared_buff, intermediate_key_shared_buff, totalKeySize);
	memcpy(sorted_vals_shared_buff, intermediate_val_shared_buff, totalValSize);

	int sorted_key_arr_len = 0;

	///////////////////////////////////////////////////////////////////////////////////////////////////
	//transfer the d_sorted_keyval_pos_arr to h_sorted_keyval_pos_arr

	sorted_keyval_pos_t * h_sorted_keyval_pos_arr = NULL;
	for (int i=0; i<total_count; i++){
		int iKeySize = h_intermediate_keyvals_pos_arr[i].keySize;

		int j = 0;
		for (; j<sorted_key_arr_len; j++){

			int jKeySize = h_sorted_keyval_pos_arr[j].keySize;
			char *key_i = (char *)(intermediate_key_shared_buff + h_intermediate_keyvals_pos_arr[i].keyPos);
			char *key_j = (char *)(sorted_keys_shared_buff + h_sorted_keyval_pos_arr[j].keyPos);
			if (cpu_compare(key_i,iKeySize,key_j,jKeySize)!=0)
				continue;

			//found the match
			int arr_len = h_sorted_keyval_pos_arr[j].val_arr_len;
			h_sorted_keyval_pos_arr[j].val_pos_arr = (val_pos_t *)realloc(h_sorted_keyval_pos_arr[j].val_pos_arr, sizeof(val_pos_t)*(arr_len+1));
			h_sorted_keyval_pos_arr[j].val_pos_arr[arr_len].valSize = h_intermediate_keyvals_pos_arr[i].valSize;
			h_sorted_keyval_pos_arr[j].val_pos_arr[arr_len].valPos = h_intermediate_keyvals_pos_arr[i].valPos;
			h_sorted_keyval_pos_arr[j].val_arr_len ++;
			break;
		}//for

		if(j==sorted_key_arr_len){
			sorted_key_arr_len++;
			h_sorted_keyval_pos_arr = (sorted_keyval_pos_t *)realloc(h_sorted_keyval_pos_arr,sorted_key_arr_len*sizeof(sorted_keyval_pos_t));
			sorted_keyval_pos_t *p = &(h_sorted_keyval_pos_arr[sorted_key_arr_len - 1]);
			p->keySize = iKeySize;
			p->keyPos = h_intermediate_keyvals_pos_arr[i].keyPos;

			p->val_arr_len = 1;
			p->val_pos_arr = (val_pos_t*)malloc(sizeof(val_pos_t));
			p->val_pos_arr[0].valSize = h_intermediate_keyvals_pos_arr[i].valSize;
			p->val_pos_arr[0].valPos = h_intermediate_keyvals_pos_arr[i].valPos;
		}//if

	}
	pgc->sorted_key_vals.h_sorted_keyval_pos_arr	= h_sorted_keyval_pos_arr;
	pgc->sorted_key_vals.d_sorted_keyvals_arr_len	= sorted_key_arr_len;

	keyval_pos_t *tmp_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
	//ShowLog("GPU_ID:[%d] #input_records:%d #intermediate_records:%lu #different_intermediate_records:%d totalKeySize:%d KB totalValSize:%d KB", 
	//	d_g_state->gpu_id, d_g_state->num_input_record, total_count, sorted_key_arr_len,totalKeySize/1024,totalValSize/1024);

	int *pos_arr_4_pos_arr = (int*)malloc(sizeof(int)*sorted_key_arr_len);
	memset(pos_arr_4_pos_arr,0,sizeof(int)*sorted_key_arr_len);

	int	index = 0;
	for (int i=0;i<sorted_key_arr_len;i++){
		sorted_keyval_pos_t *p = (sorted_keyval_pos_t *)&(h_sorted_keyval_pos_arr[i]);

		for (int j=0;j<p->val_arr_len;j++){
			tmp_keyval_pos_arr[index].keyPos = p->keyPos;
			tmp_keyval_pos_arr[index].keySize = p->keySize;
			tmp_keyval_pos_arr[index].valPos = p->val_pos_arr[j].valPos;
			tmp_keyval_pos_arr[index].valSize = p->val_pos_arr[j].valSize;
			index++;
		}//for
		pos_arr_4_pos_arr[i] = index;

	}
	cudaMemcpy(pgc->sorted_key_vals.d_keyval_pos_arr,tmp_keyval_pos_arr,sizeof(keyval_pos_t)*total_count,cudaMemcpyHostToDevice);
	pgc->sorted_key_vals.d_sorted_keyvals_arr_len = sorted_key_arr_len;
	cudaMalloc((void**)&pgc->sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr,sizeof(int)*sorted_key_arr_len);
	cudaMemcpy(pgc->sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr,pos_arr_4_pos_arr,sizeof(int)*sorted_key_arr_len,cudaMemcpyHostToDevice);

}

void ExecutePandaCPUSort(panda_cpu_context *pcc, panda_node_context *pnc){

	//keyvals_t * merged_keyvals_arr = NULL;
	int num_threads = pcc->num_cpus_cores > pcc->input_key_vals.num_input_record ? pcc->input_key_vals.num_input_record:pcc->num_cpus_cores;
	int num_records_per_thread = (pcc->input_key_vals.num_input_record)/(num_threads);

	ShowLog("num cores:%d num input records:%d threads:%d num_reocords_per_thread:%d",
			pcc->num_cpus_cores, pcc->input_key_vals.num_input_record, num_threads,num_records_per_thread);

	int start_idx = 0;
	int end_idx = 0;
	
	int total_count = 0;
	for (int i=0; i< pcc->input_key_vals.num_input_record; i++){
		total_count += pcc->intermediate_key_vals.intermediate_keyval_total_count[i];
	}//for

	int keyvals_arr_max_len = pnc->sorted_key_vals.sorted_keyval_arr_max_len;
	//pnc->sorted_key_vals.sorted_intermediate_keyvals_arr = (keyvals_t *)malloc(sizeof(keyvals_t)*keyvals_arr_len);
	keyvals_t * sorted_intermediate_keyvals_arr = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr;
			
	int sorted_key_arr_len = 0;

	for (int tid = 0;tid<num_threads;tid++){
	
		end_idx = start_idx + num_records_per_thread;
		if (tid < (pcc->input_key_vals.num_input_record % num_threads) )
			end_idx++;
		if (end_idx > pcc->input_key_vals.num_input_record)
			end_idx = pcc->input_key_vals.num_input_record;
		if (end_idx<=start_idx) continue;
		keyval_arr_t *kv_arr_p 	= (keyval_arr_t *)&(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[start_idx]);
		int shared_arr_len 	= *kv_arr_p->shared_arr_len;
		char *shared_buff 	= kv_arr_p->shared_buff;
		int shared_buff_len 	= *kv_arr_p->shared_buff_len;
		//bool local_combiner 	= pnc->local_combiner;
		bool local_combiner 	= false;

		for(int local_idx = 0; local_idx<(shared_arr_len); local_idx++){

			keyval_pos_t *p2 = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len - local_idx ));
			if (local_combiner && p2->next_idx != _COMBINE)		continue;
		
			char *key_i = shared_buff + p2->keyPos;
			char *val_i = shared_buff + p2->valPos;
			ShowLog("cihui key:%s val:%d",key_i,*(int *)(val_i));
			int keySize_i = p2->keySize;
			int valSize_i = p2->valSize;
		
			int k = 0;
			for (; k<sorted_key_arr_len; k++){
				char *key_k = (char *)(sorted_intermediate_keyvals_arr[k].key);
				int keySize_k = sorted_intermediate_keyvals_arr[k].keySize;

				if ( cpu_compare(key_i, keySize_i, key_k, keySize_k) != 0 )
					continue;

				//found the match
				val_t *vals = sorted_intermediate_keyvals_arr[k].vals;
				sorted_intermediate_keyvals_arr[k].val_arr_len++;
			sorted_intermediate_keyvals_arr[k].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[k].val_arr_len));

				int index = sorted_intermediate_keyvals_arr[k].val_arr_len - 1;
				sorted_intermediate_keyvals_arr[k].vals[index].valSize = valSize_i;
				sorted_intermediate_keyvals_arr[k].vals[index].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(sorted_intermediate_keyvals_arr[k].vals[index].val,val_i,valSize_i);
				break;

			}//for
			
			if (k == sorted_key_arr_len){
				sorted_key_arr_len++;
				if (sorted_key_arr_len >= keyvals_arr_max_len){

					keyvals_arr_max_len*=2;
					keyvals_t* new_sorted_intermediate_keyvals_arr = (keyvals_t *)malloc(sizeof(keyvals_t)*keyvals_arr_max_len);
					memcpy(new_sorted_intermediate_keyvals_arr, sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*keyvals_arr_max_len/2);
					sorted_intermediate_keyvals_arr=new_sorted_intermediate_keyvals_arr;

				}//if

				//sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*sorted_key_arr_len);
				int index = sorted_key_arr_len-1;
				keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);
				kvals_p->keySize = keySize_i;

				kvals_p->key = malloc(sizeof(char)*keySize_i);
				memcpy(kvals_p->key, key_i, keySize_i);

				kvals_p->vals = (val_t *)malloc(sizeof(val_t));
				kvals_p->val_arr_len = 1;

				kvals_p->vals[0].valSize = valSize_i;
				kvals_p->vals[0].val = (char *)malloc(sizeof(char)*valSize_i);
				memcpy(kvals_p->vals[0].val,val_i, valSize_i);
				pnc->sorted_key_vals.sorted_keyval_arr_max_len = keyvals_arr_max_len;

			}//if
		}
	
		free(shared_buff);
		start_idx = end_idx;
		pnc->sorted_key_vals.sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;
	}
	pnc->sorted_key_vals.sorted_keyvals_arr_len = sorted_key_arr_len;

}

void ExecutePandaGPUShuffleMerge(panda_node_context *pnc, panda_gpu_context *pgc){

	char *sorted_keys_shared_buff_0 = (char *)pgc->sorted_key_vals.h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff_0 = (char *)pgc->sorted_key_vals.h_sorted_vals_shared_buff;

	sorted_keyval_pos_t *keyval_pos_arr_0 = pgc->sorted_key_vals.h_sorted_keyval_pos_arr;
	keyvals_t *sorted_intermediate_keyvals_arr = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr;

	void *key_0, *key_1;
	int keySize_0, keySize_1;
	//bool equal;

	//int new_count = 0;
	for (int i=0;i< pgc->sorted_key_vals.d_sorted_keyvals_arr_len;i++){
		key_0 = sorted_keys_shared_buff_0 + keyval_pos_arr_0[i].keyPos;
		keySize_0 = keyval_pos_arr_0[i].keySize;
		int j = 0;
		
		for (; j< pnc->sorted_key_vals.sorted_keyvals_arr_len; j++){

			key_1 = sorted_intermediate_keyvals_arr[j].key;
			keySize_1 = sorted_intermediate_keyvals_arr[j].keySize;

			if(cpu_compare(key_0,keySize_0,key_1,keySize_1)!=0)
				continue;

			val_t *vals = sorted_intermediate_keyvals_arr[j].vals;
			//copy values from gpu to cpu context
			int val_arr_len_0 = keyval_pos_arr_0[i].val_arr_len;
			val_pos_t * val_pos_arr = keyval_pos_arr_0[i].val_pos_arr;

			int index = sorted_intermediate_keyvals_arr[j].val_arr_len;
			sorted_intermediate_keyvals_arr[j].val_arr_len += val_arr_len_0;
			sorted_intermediate_keyvals_arr[j].vals = (val_t*)realloc(vals, sizeof(val_t)*(sorted_intermediate_keyvals_arr[j].val_arr_len));

			for (int k=0; k < val_arr_len_0; k++){

				char *val_0 = sorted_vals_shared_buff_0 + val_pos_arr[k].valPos;
				int valSize_0 = val_pos_arr[k].valSize;

				sorted_intermediate_keyvals_arr[j].vals[index+k].val = (char *)malloc(sizeof(char)*valSize_0);
				sorted_intermediate_keyvals_arr[j].vals[index+k].valSize = valSize_0;
				memcpy(sorted_intermediate_keyvals_arr[j].vals[index+k].val, val_0, valSize_0);

			}//for
			break;
		}//for

		if (j == pnc->sorted_key_vals.sorted_keyvals_arr_len){

			if (pnc->sorted_key_vals.sorted_keyvals_arr_len == 0) sorted_intermediate_keyvals_arr = NULL;
			int val_arr_len =keyval_pos_arr_0[i].val_arr_len;
			val_pos_t * val_pos_arr =keyval_pos_arr_0[i].val_pos_arr;
			pnc->sorted_key_vals.sorted_keyvals_arr_len++;

			sorted_intermediate_keyvals_arr = (keyvals_t *)realloc(sorted_intermediate_keyvals_arr, sizeof(keyvals_t)*(pnc->sorted_key_vals.sorted_keyvals_arr_len));
			int index = pnc->sorted_key_vals.sorted_keyvals_arr_len-1;
			keyvals_t* kvals_p = (keyvals_t *)&(sorted_intermediate_keyvals_arr[index]);

			kvals_p->keySize = keySize_0;
			kvals_p->key = malloc(sizeof(char)*keySize_0);

			memcpy(kvals_p->key, key_0, keySize_0);
			kvals_p->vals = (val_t *)malloc(sizeof(val_t)*val_arr_len);
			kvals_p->val_arr_len = val_arr_len;

			for (int k=0; k < val_arr_len; k++){

				char *val_0 = sorted_vals_shared_buff_0 + val_pos_arr[k].valPos;
				int valSize_0 = val_pos_arr[k].valSize;

				kvals_p->vals[k].valSize = valSize_0;
				kvals_p->vals[k].val = (char *)malloc(sizeof(char)*valSize_0);
				memcpy(kvals_p->vals[k].val, val_0, valSize_0);

			}//for
		}//if
	}//if 
	pnc->sorted_key_vals.sorted_intermediate_keyvals_arr = sorted_intermediate_keyvals_arr;

	return;

}//void

__global__ void copyDataFromDevice2Host1(panda_gpu_context pgc)
{
	
	int num_records_per_thread = (pgc.input_key_vals.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);

	printf("[copyDataFromDevice2Host1]  pgc.input_key_vals.num_input_record:%d gx:%d,bx:%d,by:%d\n",
		pgc.input_key_vals.num_input_record,gridDim.x,blockDim.x,blockDim.y);

	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread;//*STRIDE;

	if(thread_end_idx>pgc.input_key_vals.num_input_record)
		thread_end_idx = pgc.input_key_vals.num_input_record;

	//printf("thread_start_idx:%d  thread_end_idx:%d\n",thread_start_idx,thread_end_idx);

	if (thread_start_idx >= thread_end_idx)
		return;

	printf("thread_start_idx:%d  thread_end_idx:%d\n",thread_start_idx,thread_end_idx);

	int begin=0;
	int end=0;
	for (int i=0; i<thread_start_idx; i++){
		begin += pgc.intermediate_key_vals.d_intermediate_keyval_total_count[i];
	}//for
	end = begin + pgc.intermediate_key_vals.d_intermediate_keyval_total_count[thread_start_idx];

	printf("begin:%d end:%d\n",begin,end);
	
	int start_idx = 0;
	//bool local_combiner = d_g_state.local_combiner;
	bool local_combiner = false;

	for(int i=begin;i<end;i++){
		keyval_t * p1 = &(pgc.intermediate_key_vals.d_intermediate_keyval_arr[i]);
		keyval_pos_t * p2 = NULL;
		keyval_arr_t *kv_arr_p = pgc.intermediate_key_vals.d_intermediate_keyval_arr_arr_p[thread_start_idx];

		char *shared_buff = (char *)(kv_arr_p->shared_buff);
		int shared_arr_len = *kv_arr_p->shared_arr_len;
		int shared_buff_len = *kv_arr_p->shared_buff_len;

		for (int idx = start_idx; idx<(shared_arr_len); idx++){
	p2 = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len - idx ));
			if ( local_combiner && p2->next_idx != _COMBINE ){
				continue;
			}//if

			start_idx = idx+1;

			p1->keySize = p2->keySize;
			p1->valSize = p2->valSize;
			p1->task_idx = i;
			p2->task_idx = i;
			break;
		}//for
	}//for
}

__global__ void copyDataFromDevice2Host2(panda_gpu_context pgc)
{	

	int num_records_per_thread = (pgc.input_key_vals.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx+num_records_per_thread;

	if(thread_end_idx>pgc.input_key_vals.num_input_record)
		thread_end_idx = pgc.input_key_vals.num_input_record;

	if (thread_start_idx >= thread_end_idx)
		return;

	int begin=0;
	int end=0;
	for (int i=0; i<thread_start_idx; i++) 	
		begin = begin + pgc.intermediate_key_vals.d_intermediate_keyval_total_count[i];

	end = begin + pgc.intermediate_key_vals.d_intermediate_keyval_total_count[thread_start_idx];

	keyval_arr_t *kv_arr_p = pgc.intermediate_key_vals.d_intermediate_keyval_arr_arr_p[thread_start_idx];
	char *shared_buff = (char *)(kv_arr_p->shared_buff);
	int shared_arr_len = *kv_arr_p->shared_arr_len;
	int shared_buff_len = *kv_arr_p->shared_buff_len;

	int val_pos, key_pos;
	char *val_p,*key_p;
	//int counter = 0;
	//bool local_combiner = d_g_state.local_combiner;
	bool local_combiner = false;

	for(int local_idx = 0; local_idx<(shared_arr_len); local_idx++){

	keyval_pos_t *p2 = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len - local_idx ));
	if (local_combiner && p2->next_idx != _COMBINE)		continue;
	//	if (p2->task_idx != i) 		continue;
	int global_idx = p2->task_idx;
	val_pos = pgc.intermediate_key_vals.d_intermediate_keyval_pos_arr[global_idx].valPos;
	key_pos = pgc.intermediate_key_vals.d_intermediate_keyval_pos_arr[global_idx].keyPos;

	val_p = (char*)(pgc.intermediate_key_vals.d_intermediate_vals_shared_buff)+val_pos;
	key_p = (char*)(pgc.intermediate_key_vals.d_intermediate_keys_shared_buff)+key_pos;

	memcpy(key_p, shared_buff + p2->keyPos, p2->keySize);
	memcpy(val_p, shared_buff + p2->valPos, p2->valSize);

	//counter++;
	}

	//if(counter!=end-begin)
	//	ShowWarn("counter!=end-begin counter:%d end-begin:%d begin:%d end:%d shared_arr_len:%d",counter,end-begin,begin,end,shared_arr_len);
	
	free(shared_buff);

}//__global__	

 inline int cpu_compare(const void *key_a,int len_a, const void *key_b,int len_b){
        int short_len = len_a>len_b? len_b:len_a;
        for(int i = 0;i<short_len;i++){
                if(((char *)key_a)[i]>((char *)key_b)[i])
                return 1;
                if(((char *)key_a)[i]<((char *)key_b)[i])
                return -1;
        }
	if(len_a>len_b) return 1;
	else if(len_a<len_b) return -1;
	else if(len_a == len_b) return 0;
  }

 __device__ int gpu_compare(const void *key_a, int len_a, const void *key_b, int len_b){
    int short_len = len_a>len_b? len_b:len_a;
    for(int i = 0;i<short_len;i++){
        if(((char *)key_a)[i]>((char *)key_b)[i])
        return 1;
        if(((char *)key_a)[i]<((char *)key_b)[i])
        return -1;
    }
    if(len_a>len_b) return 1;
    else if(len_a<len_b) return -1;
    else if(len_a == len_b) return 0;
  }

#endif 
