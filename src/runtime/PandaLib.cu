/*
	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	Panda: co-processing SPMD computations on GPUs and CPUs.
	
	File: PandaLib.cu
	First Version:		2012-07-01 V0.1
	Last UPdates: 		2018-04-28 v0.61	
	Developer: Hui Li (huili@ruijie.com.cn)

 */

//#ifndef __PANDA_LIB_CU__
//#define __PANDA_LIB_CU__

#include "Panda.h"
#include "PandaAPI.h"
#include <stdio.h>
#include <cstdio>
#include <cstdlib>

namespace panda{

extern int gCommRank;

__global__ void RunPandaGPUMapTasksSplit(panda_gpu_context pgc)
{
	//ShowLog2("lihuix9 gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d blockIdx.x:%d blockIdx.y:%d blockIdx.z:%d\n",
	// 		gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z,blockIdx.x,blockIdx.y,blockIdx.z);
	int num_records_per_thread = (pgc.input_key_vals.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;
	if (thread_end_idx > pgc.input_key_vals.num_input_record)
		thread_end_idx = pgc.input_key_vals.num_input_record;

	if (thread_start_idx >= thread_end_idx)
		return;

	printf("lihuix9 num_records_per_thread:%d\n",num_records_per_thread);

	int buddy_arr_len = num_records_per_thread;
	int *int_arr = (int*)malloc((4+buddy_arr_len)*sizeof(int));
	if(int_arr==NULL){ GpuErrorLog("there is not enough GPU memory\n"); return;}

	int *shared_arr_len = int_arr;
	int *shared_buff_len = int_arr+1;
	int *shared_buff_pos = int_arr+2;
	//int *num_buddy = int_arr+3;
	int *buddy = int_arr+4;
	(*shared_buff_len) = SHARED_BUFF_LEN;
	(*shared_arr_len) = 0;
	(*shared_buff_pos) = 0;

	char * buff = (char *)malloc(sizeof(char)*(*shared_buff_len));
	keyval_arr_t *kv_arr_t_arr = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*(thread_end_idx-thread_start_idx+STRIDE-1)/STRIDE);
	int index = 0;
	
	for(int idx=thread_start_idx;idx<thread_end_idx;idx+=STRIDE){
			buddy[index]=idx;
			index++;
	}//for
	index = 0;
	for(int map_task_idx = thread_start_idx; map_task_idx < thread_end_idx; map_task_idx += STRIDE){

		keyval_arr_t *kv_arr_t = (keyval_arr_t *)&(kv_arr_t_arr[index]);
		index++;
		kv_arr_t->shared_buff = buff;
		kv_arr_t->shared_arr_len = shared_arr_len;
		kv_arr_t->shared_buff_len = shared_buff_len;
		kv_arr_t->shared_buff_pos = shared_buff_pos;
		kv_arr_t->shared_buddy = buddy;
		kv_arr_t->shared_buddy_len = buddy_arr_len;
		kv_arr_t->arr = NULL;
		kv_arr_t->arr_len = 0;
		
		pgc.intermediate_key_vals.d_intermediate_keyval_arr_arr_p[map_task_idx] = kv_arr_t;

	}//for
}

void ExecutePandaGPUMapTasksSplit(panda_gpu_context pgc, dim3 grids, dim3 blocks)
{
   	RunPandaGPUMapTasksSplit<<<grids,blocks>>>(pgc);
}

void* RunPandaCPUMapThread(void * ptr)
{

	panda_cpu_task_info_t *panda_cpu_task_info = (panda_cpu_task_info_t *)ptr;
	panda_cpu_context  *pcc = (panda_cpu_context *) (panda_cpu_task_info->pcc);
	//panda_node_context *pnc = (panda_node_context *)(panda_cpu_task_info->pnc);
	
	int start_task_idx	=	panda_cpu_task_info->start_task_idx;
	int end_task_idx	=	panda_cpu_task_info->end_task_idx;

	if(end_task_idx<=start_task_idx) 	return NULL;
	
	char *buff		=	(char *)malloc(sizeof(char)*CPU_SHARED_BUFF_SIZE);
	int *int_arr	=	(int *)malloc(sizeof(int)*(end_task_idx - start_task_idx + 3));
	int *buddy		=	int_arr+3;
	
	int buddy_len	=	end_task_idx	- start_task_idx;
	for (int i=0;i<buddy_len;i++){
		buddy [i]	=	i + start_task_idx;
	}//for
	
	for (int map_idx = start_task_idx; map_idx < end_task_idx; map_idx++){

		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff)		= buff;
		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_len) = int_arr;
		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_pos) = int_arr+1;
		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_arr_len)	= int_arr+2;
		
		*(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_len)	= CPU_SHARED_BUFF_SIZE;
		*(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_pos)	= 0;
		*(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_arr_len)		= 0;
		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buddy)		= buddy;
		(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buddy_len)	= buddy_len;

	}//for

	for (int map_idx = panda_cpu_task_info->start_task_idx; map_idx < panda_cpu_task_info->end_task_idx; map_idx++){
		keyval_t *kv_p = (keyval_t *)(&(pcc->input_key_vals.input_keyval_arr[map_idx]));
		panda_cpu_map(kv_p->key,kv_p->val,kv_p->keySize,kv_p->valSize,pcc,map_idx);
	}//for
	
	//ShowLog("CPU_GROUP_ID:[%d] Done :%d tasks",d_g_state->cpu_group_id, panda_cpu_task_info->end_task_idx - panda_cpu_task_info->start_task_idx);
	return NULL;
}//int 


void ExecutePandaGPUReduceTasks(panda_gpu_context *pgc)
{
	if (pgc->sorted_key_vals.d_sorted_keyvals_arr_len <= 0)
	return;

	cudaThreadSynchronize(); 
	pgc->reduced_key_vals.d_reduced_keyval_arr_len = pgc->sorted_key_vals.d_sorted_keyvals_arr_len;

	cudaMalloc((void **)&(pgc->reduced_key_vals.d_reduced_keyval_arr), 
		sizeof(keyval_t)*pgc->reduced_key_vals.d_reduced_keyval_arr_len);

	pgc->output_key_vals.totalKeySize = 0;
	pgc->output_key_vals.totalValSize = 0;
	pgc->output_key_vals.h_reduced_keyval_arr_len = pgc->reduced_key_vals.d_reduced_keyval_arr_len;
	pgc->output_key_vals.h_reduced_keyval_arr = (keyval_t*)(malloc(sizeof(keyval_t)*pgc->output_key_vals.h_reduced_keyval_arr_len));
	
	cudaThreadSynchronize();
	int numGPUCores = pgc->num_gpus_cores;
	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
        dim3 grids(numBlocks, 1);
	//int total_gpu_threads = (grids.x*grids.y*blocks.x*blocks.y);
	//pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_len = pgc->reduced_key_vals.d_reduced_keyval_arr_len;
	
	ShowLog("reduced len:%d output len:%d sorted keySize%d: sorted valSize:%d",
		pgc->reduced_key_vals.d_reduced_keyval_arr_len, 
		pgc->output_key_vals.h_reduced_keyval_arr_len,
		pgc->sorted_key_vals.totalKeySize, 
		pgc->sorted_key_vals.totalValSize);

	RunPandaGPUReduceTasks<<<grids,blocks>>>(*pgc);

	cudaMemcpy(pgc->output_key_vals.h_reduced_keyval_arr,
		pgc->reduced_key_vals.d_reduced_keyval_arr,
		sizeof(keyval_t)*pgc->reduced_key_vals.d_reduced_keyval_arr_len,
		cudaMemcpyDeviceToHost);

	for (int i = 0; i<pgc->reduced_key_vals.d_reduced_keyval_arr_len; i++){
		pgc->output_key_vals.totalKeySize += (pgc->output_key_vals.h_reduced_keyval_arr[i].keySize+3)/4*4;
		pgc->output_key_vals.totalValSize += (pgc->output_key_vals.h_reduced_keyval_arr[i].valSize+3)/4*4;
	}//for
	
	//ShowLog("Output total keySize:%f KB valSize:%f KB\n",(float)(pgc->output_key_vals.totalKeySize)/1024.0,(float)(pgc->output_key_vals.totalValSize)/1024.0);

	pgc->output_key_vals.h_KeyBuff = malloc(sizeof(char)*pgc->output_key_vals.totalKeySize);
	pgc->output_key_vals.h_ValBuff = malloc(sizeof(char)*pgc->output_key_vals.totalValSize);

	cudaMalloc(&(pgc->output_key_vals.d_KeyBuff), sizeof(char)*pgc->output_key_vals.totalKeySize );
	cudaMalloc(&(pgc->output_key_vals.d_ValBuff), sizeof(char)*pgc->output_key_vals.totalValSize );

	ShowLog("[copyDataFromDevice2Host4Reduce] Output total keySize:%f KB valSize:%f KB\n",(float)(pgc->output_key_vals.totalKeySize)/1024.0,(float)(pgc->output_key_vals.totalValSize)/1024.0);

	copyDataFromDevice2Host4Reduce<<<grids,blocks>>>(*pgc);

	cudaMemcpy(
			pgc->output_key_vals.h_KeyBuff,
			pgc->output_key_vals.d_KeyBuff,
			pgc->output_key_vals.totalKeySize,
		cudaMemcpyDeviceToHost);

	cudaMemcpy(
		pgc->output_key_vals.h_ValBuff,
		pgc->output_key_vals.d_ValBuff,
		pgc->output_key_vals.totalValSize,
		cudaMemcpyDeviceToHost);

	int val_pos, key_pos;
	val_pos = key_pos = 0;
	void *val, *key;

	for (int i = 0; i<pgc->output_key_vals.h_reduced_keyval_arr_len; i++){
		
		val = (char *)pgc->output_key_vals.h_ValBuff + val_pos;
		key = (char *)pgc->output_key_vals.h_KeyBuff + key_pos;
		pgc->output_key_vals.h_reduced_keyval_arr[i].key = key;
		pgc->output_key_vals.h_reduced_keyval_arr[i].val = val;
		ShowLog("key:%s val:%d",(char *)key,*(int*)val);

		val_pos += (pgc->output_key_vals.h_reduced_keyval_arr[i].valSize+3)/4*4;
		key_pos += (pgc->output_key_vals.h_reduced_keyval_arr[i].keySize+3)/4*4;

	}//for

	//TODO
	cudaThreadSynchronize(); 

}//void

__global__ void RunPandaGPUReduceTasks(panda_gpu_context pgc)
{
	//ErrorLog("ReducePartitioner Panda_GPU_Context");
	int num_records_per_thread = (pgc.sorted_key_vals.d_sorted_keyvals_arr_len + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);

	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread;//*STRIDE;
	
	if (thread_end_idx > pgc.sorted_key_vals.d_sorted_keyvals_arr_len)
		thread_end_idx = pgc.sorted_key_vals.d_sorted_keyvals_arr_len;

	if (thread_start_idx >= thread_end_idx)
		return;

	int start_idx, end_idx;
	for(int reduce_task_idx=thread_start_idx; reduce_task_idx < thread_end_idx; reduce_task_idx++/*=STRIDE*/){

		if (reduce_task_idx==0)
			start_idx = 0;
		else
			start_idx = pgc.sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx-1];
		end_idx = pgc.sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr[reduce_task_idx];
		val_t *val_t_arr = (val_t*)malloc(sizeof(val_t)*(end_idx-start_idx));
		
		int keySize = pgc.sorted_key_vals.d_keyval_pos_arr[start_idx].keySize;
		int keyPos = pgc.sorted_key_vals.d_keyval_pos_arr[start_idx].keyPos;
		void *key = (char*)pgc.sorted_key_vals.d_sorted_keys_shared_buff+keyPos;
				
		for (int index = start_idx;index<end_idx;index++){
			int valSize = pgc.sorted_key_vals.d_keyval_pos_arr[index].valSize;
			int valPos = pgc.sorted_key_vals.d_keyval_pos_arr[index].valPos;
			val_t_arr[index-start_idx].valSize = valSize;
			val_t_arr[index-start_idx].val = (char*)pgc.sorted_key_vals.d_sorted_vals_shared_buff + valPos;
		}   //for
		if( end_idx - start_idx == 0) {
		GpuErrorLog("gpu_reduce valCount ==0");
		}//if
		else panda_gpu_core_reduce(key, val_t_arr, keySize, end_idx-start_idx, pgc);
	}//for
}

__global__ void RunPandaGPUMapTasksIterative(panda_gpu_context pgc, int curIter, int totalIter)
{
	
	if(blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
	{
	printf("lihuix 5 gridDim.x:%u gridDim.y:%u gridDim.z:%u blockIdx.x:%u blockIdx.y:%u blockIdx.z:%u threadIdx.x:%u threadIdx.y:%u threadIdx.z:%u\n",
	  	gridDim.x,gridDim.y,gridDim.z,blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z);
	}
	int num_records_per_thread = (pgc.input_key_vals.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);
	int thread_end_idx = thread_start_idx + num_records_per_thread*STRIDE;
	if (thread_end_idx > pgc.input_key_vals.num_input_record)
		thread_end_idx = pgc.input_key_vals.num_input_record;
	if(blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
	{
	printf("lihuix 6 num_records_per_thread:%d block_start_idx:%u thread_start_idx:%d thread_end_idx:%d STRIDE:%d\n",num_records_per_thread, block_start_idx, thread_start_idx, thread_end_idx, STRIDE);
	}//if

	if (thread_start_idx + curIter*STRIDE >= thread_end_idx)
		return;

	for(int map_task_idx = thread_start_idx + curIter*STRIDE; map_task_idx < thread_end_idx; map_task_idx += totalIter*STRIDE){

		char *key = (char *)(pgc.input_key_vals.d_input_keys_shared_buff) + pgc.input_key_vals.d_input_keyval_pos_arr[map_task_idx].keyPos;
		char *val = (char *)(pgc.input_key_vals.d_input_vals_shared_buff) + pgc.input_key_vals.d_input_keyval_pos_arr[map_task_idx].valPos;
		//if(blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		//printf("lihuix11 map_task_idx:%d key:%d  keyPos:%d \n",map_task_idx, *((int *)key),pgc.input_key_vals.d_input_keyval_pos_arr[map_task_idx].keyPos);
		int valSize = pgc.input_key_vals.d_input_keyval_pos_arr[map_task_idx].valSize;
		int keySize = pgc.input_key_vals.d_input_keyval_pos_arr[map_task_idx].keySize;

		printf("lihuix11 map_task_idx:%d key:%d  keyPos:%d valSize:%d\n",map_task_idx, *((int *)key),pgc.input_key_vals.d_input_keyval_pos_arr[map_task_idx].keyPos,valSize);
		/////////////////////////////////////////////////////////////////////
		panda_gpu_core_map(key, val, keySize, valSize, &pgc, map_task_idx);//
		/////////////////////////////////////////////////////////////////////
	}//for

	keyval_arr_t *kv_arr_p = pgc.intermediate_key_vals.d_intermediate_keyval_arr_arr_p[thread_start_idx];
	//char *shared_buff = (char *)(kv_arr_p->shared_buff);
	//int shared_arr_len = *kv_arr_p->shared_arr_len;
	//int shared_buff_len = *kv_arr_p->shared_buff_len;
	pgc.intermediate_key_vals.d_intermediate_keyval_total_count[thread_start_idx] = *kv_arr_p->shared_arr_len;
	printf("CUDA Debug thread_start_idx:%d  total_count:%d\n",thread_start_idx,*kv_arr_p->shared_arr_len);
	__syncthreads();
}//GPUMapPartitioner

void *RunPandaCPUCombinerThread(void *ptr){

	panda_cpu_task_info_t *panda_cpu_task_info = (panda_cpu_task_info_t *)ptr;
	panda_cpu_context *pcc = (panda_cpu_context *)(panda_cpu_task_info->pcc); 
	bool local_combiner = false;

	int start_idx = panda_cpu_task_info->start_task_idx;
	keyval_arr_t *kv_arr_p = (keyval_arr_t *)&(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[start_idx]);
	int unmerged_shared_arr_len = *kv_arr_p->shared_arr_len;

	ShowLog("lihuix unmerged_shared_arr_len:%d",unmerged_shared_arr_len);
       
	char *shared_buff = kv_arr_p->shared_buff;
        int shared_buff_len = *kv_arr_p->shared_buff_len;

	val_t *val_t_arr = (val_t *)malloc(sizeof(val_t)*unmerged_shared_arr_len);
	if (val_t_arr == NULL) ErrorLog("there is no enough memory");
	int num_keyval_pairs_after_combiner = 0;
	int total_intermediate_keyvalue_pairs = 0;

	//ShowLog("Thread unmerged_shared_arr_len:%d start_idx:%d",unmerged_shared_arr_len,start_idx);

	for (int i = 0; i < unmerged_shared_arr_len; i++){

		keyval_pos_t *head_kv_p = (keyval_pos_t *)(shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(unmerged_shared_arr_len-i));
		keyval_pos_t *first_kv_p = head_kv_p;

		if (first_kv_p->next_idx != _MAP)
			continue;

		int iKeySize = first_kv_p->keySize;
		char *iKey = shared_buff + first_kv_p->keyPos;
		//char *iVal = shared_buff + first_kv_p->valPos;
		if((first_kv_p->keyPos%4!=0)||(first_kv_p->valPos%4!=0)){
			ErrorLog("keyPos or valPos is not aligned with 4 bytes, results could be wrong");
		}
	
		int index = 0;
		(val_t_arr[index]).valSize = first_kv_p->valSize;
		(val_t_arr[index]).val = (char*)shared_buff + first_kv_p->valPos;

		//ShowLog("unmerged_shared_arr_len:%d combiner key:%s val:%d",unmerged_shared_arr_len,iKey,*(int *)(val_t_arr[index]).val);
		//printf("cominber key:");
		//for(int s = 0;s<10;s++)
	//		printf("%3d",(int)iKey[s]);
	//	printf("\n");
		for (int j=i+1;j<unmerged_shared_arr_len;j++){

			keyval_pos_t *next_kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(unmerged_shared_arr_len-j));
			char *jKey = (char *)shared_buff+next_kv_p->keyPos;
			int jKeySize = next_kv_p->keySize;
		
			if (!local_combiner||cpu_compare(iKey,iKeySize,jKey,jKeySize)!=0){
				continue;
			}
			index++;
			first_kv_p->next_idx = j;
			first_kv_p = next_kv_p;
			(val_t_arr[index]).valSize = next_kv_p->valSize;
			(val_t_arr[index]).val = (char*)shared_buff + next_kv_p->valPos;
		}

		int valCount = index+1;
		total_intermediate_keyvalue_pairs += valCount;
		if(valCount>1){
			panda_cpu_combiner(iKey,val_t_arr,iKeySize,(valCount),pcc,start_idx);
		}//int
		else{
			first_kv_p->next_idx = _COMBINE;
			first_kv_p->task_idx = start_idx;
		}
		num_keyval_pairs_after_combiner++;
	}//for
	free(val_t_arr);
	pcc->intermediate_key_vals.intermediate_keyval_total_count[start_idx] = num_keyval_pairs_after_combiner;
	/*
	ShowLog("CPU_GROUP_ID:[%d] Map_Idx:%d  Done:%d Combiner: %d => %d Compress Ratio:%f",
		d_g_state->cpu_group_id, 
		panda_cpu_task_info->start_task_idx,
		panda_cpu_task_info->end_task_idx - panda_cpu_task_info->start_task_idx, 
		total_intermediate_keyvalue_pairs,
		num_keyval_pairs_after_combiner,
		(num_keyval_pairs_after_combiner/(float)total_intermediate_keyvalue_pairs)
		);
	*/
	return NULL;
}

void ExecutePandaGPUMapTasksIterative(panda_gpu_context pgc, int curIter, int totalIter, dim3 grids, dim3 blocks){
	ShowLog("lihuix2 curIter:%d totalIter:%d\n GPUMapTasksIterative",curIter,totalIter);
	RunPandaGPUMapTasksIterative<<<grids,blocks>>>(pgc, totalIter -1 - curIter, totalIter);
	ShowLog("lihuix4\n");	
}//void

void ExecutePandaGPUCombiner(panda_gpu_context * pgc){
	
	ShowLog("hello world.\n");
	//cudaMemset(pgc->intermediate_key_vals.d_intermediate_keyval_total_count,0,pgc->input_key_vals.num_input_record*sizeof(int));
	ShowLog("lihuix 13 pgc->input_key_vals.num_input_record:%d",pgc->input_key_vals.num_input_record);
	int numGPUCores = pgc->num_gpus_cores;
	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
    	dim3 grids(numBlocks, 1);

	RunPandaGPUCombiner<<<grids,blocks>>>(*pgc);
	cudaThreadSynchronize();
}

void ExecutePandaCPUCombiner(panda_cpu_context *pcc){
	if (pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p == NULL)	{ ErrorLog("intermediate_keyval_arr_arr_p == NULL"); return; }
	if (pcc->intermediate_key_vals.intermediate_keyval_arr_arr_len <= 0)	{ ErrorLog("no any input keys"); return; }
	if (pcc->num_cpus_cores <= 0)	{ ErrorLog("pcc->num_cpus == 0"); return; }
	//-------------------------------------------------------
	//1, prepare buffer to store intermediate results
	//-------------------------------------------------------

	ShowLog("num_input_record:%d",pcc->input_key_vals.num_input_record);

	int num_threads = pcc->num_cpus_cores > pcc->input_key_vals.num_input_record ? pcc->input_key_vals.num_input_record : pcc->num_cpus_cores;
	int num_records_per_thread = (pcc->input_key_vals.num_input_record)/(num_threads);
	int start_task_idx = 0;
	int end_task_idx = 0;

	for (int tid = 0;tid<num_threads;tid++){
		end_task_idx = start_task_idx + num_records_per_thread;
		if (tid < (pcc->input_key_vals.num_input_record % num_threads) )
			end_task_idx++;
		if (end_task_idx > pcc->input_key_vals.num_input_record)
			end_task_idx = pcc->input_key_vals.num_input_record;

		pcc->panda_cpu_task_thread_info[tid].start_task_idx	= start_task_idx;
		pcc->panda_cpu_task_thread_info[tid].end_task_idx	= end_task_idx;
		
		if (pthread_create(&(pcc->panda_cpu_task_thread[tid]),NULL,RunPandaCPUCombinerThread,(char *)&(pcc->panda_cpu_task_thread_info[tid]))!=0) 
			ErrorLog("Thread creation failed!");
		start_task_idx = end_task_idx;
	}//for
	
	for (int tid = 0; tid<num_threads; tid++){
		void *exitstat;
		if (pthread_join(pcc->panda_cpu_task_thread[tid],&exitstat)!=0) ErrorLog("joining failed");
	}//for
}//void


panda_gpu_context *CreatePandaGPUContext(){
	
	panda_gpu_context *pgc = (panda_gpu_context*)malloc(sizeof(panda_gpu_context));
	if (pgc == NULL) exit(-1);
	memset(pgc, 0, sizeof(panda_gpu_context));
	
	pgc->input_key_vals.d_input_keys_shared_buff = NULL;
	pgc->input_key_vals.d_input_keyval_arr = NULL;
	pgc->input_key_vals.d_input_keyval_pos_arr = NULL;
	pgc->input_key_vals.d_input_vals_shared_buff = NULL;
	pgc->input_key_vals.h_input_keyval_arr = NULL;
	pgc->input_key_vals.num_input_record = 0;
	
	pgc->intermediate_key_vals.d_intermediate_keys_shared_buff = NULL;
	pgc->intermediate_key_vals.d_intermediate_keyval_arr = NULL;
	pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_len = 0;
	pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p = NULL;
	pgc->intermediate_key_vals.d_intermediate_keyval_pos_arr = NULL;
	pgc->intermediate_key_vals.d_intermediate_keyval_total_count = 0;
	
	pgc->sorted_key_vals.d_sorted_keyvals_arr_len = 0;
	pgc->reduced_key_vals.d_reduced_keyval_arr_len = 0;
	
	return pgc;
}//gpu_context


panda_cpu_context *CreatePandaCPUContext(){
	
	panda_cpu_context *pcc = (panda_cpu_context*)malloc(sizeof(panda_cpu_context));
	if (pcc == NULL) exit(-1);
	memset(pcc, 0, sizeof(panda_cpu_context));
	
	pcc->input_key_vals.num_input_record = 0;
	pcc->input_key_vals.input_keys_shared_buff = NULL;
	pcc->input_key_vals.input_keyval_arr = NULL;
	pcc->input_key_vals.input_keyval_pos_arr = NULL;
	pcc->input_key_vals.input_vals_shared_buff = NULL;
	//pcc->input_key_vals.input_keyval_arr = NULL;
	
	pcc->intermediate_key_vals.intermediate_keys_shared_buff = NULL;
	pcc->intermediate_key_vals.intermediate_keyval_arr = NULL;
	pcc->intermediate_key_vals.intermediate_keyval_arr_arr_len = 0;
	pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p = NULL;
	pcc->intermediate_key_vals.intermediate_keyval_pos_arr = NULL;
	pcc->intermediate_key_vals.intermediate_keyval_total_count = NULL;
	
	pcc->sorted_key_vals.sorted_keyvals_arr_len = 0;
	pcc->reduced_key_vals.reduced_keyval_arr_len = 0;
	return pcc;
	
}//gpu_context

void ExecutePandaCPUReduceTasks(panda_cpu_context *pcc){

	if (pcc->sorted_key_vals.sorted_keyvals_arr_len <= 0) return;
	//keyval_t *p = (keyval_t *)(&(pcc->reduced_key_vals.reduced_keyval_arr[0]));
	pcc->reduced_key_vals.reduced_keyval_arr_len = pcc->sorted_key_vals.sorted_keyvals_arr_len;
	pcc->reduced_key_vals.reduced_keyval_arr = (keyval_t *)malloc(sizeof(keyval_t)*pcc->reduced_key_vals.reduced_keyval_arr_len);
	
	for (int reduce_idx = 0; reduce_idx < pcc->sorted_key_vals.sorted_keyvals_arr_len; reduce_idx++){
	
		keyvals_t *kv_p = (keyvals_t *)(&(pcc->sorted_key_vals.sorted_intermediate_keyvals_arr[reduce_idx]));

		if (kv_p->val_arr_len <=0) 
			ErrorLog("kv_p->val_arr_len <=0");
		else	
			panda_cpu_reduce(kv_p->key, kv_p->vals, kv_p->keySize, kv_p->val_arr_len, pcc, reduce_idx);
	}//for
		
}//void
//typedef void (*WRITE)(char *buf, void *key, void *val);
void ExecutePandaDumpReduceTasks(panda_node_context *pnc, Output *output){

	char fn[128];
	sprintf(fn,"OUTPUT_%d",gCommRank);
	FILE *fp = fopen(fn,"wb");
	ShowLog("fn:%s\n",fn);
	char *buf = NULL;
	int bs = 0;
	keyval_t *p = NULL;
	int len = pnc->output_keyval_arr.output_keyval_arr_len;
	ShowLog("len:%d\n",len);

	for(int reduce_idx=0;reduce_idx<len;reduce_idx++){

		p = (keyval_t *)(&pnc->output_keyval_arr.output_keyval_arr[reduce_idx]);
		bs = p->keySize + p->valSize;

		ShowLog("bs:%d keySize:%d   valSize:%d\n", bs, p->keySize, p->valSize);

		buf = (char *)malloc(sizeof(char)*(bs+10));
		memset(buf,0,bs+10);
		output->write(buf,p->key,p->val);
		fwrite(buf,strlen(buf),1,fp);
		free(buf);	

	}//for
	fclose(fp);
}

__global__ void copyDataFromDevice2Host4Reduce(panda_gpu_context pgc)
{

        int num_records_per_thread = (pgc.reduced_key_vals.d_reduced_keyval_arr_len
                + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
        int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;
        int thread_start_idx = block_start_idx
                + (threadIdx.y*blockDim.x + threadIdx.x)*num_records_per_thread;
        int thread_end_idx = thread_start_idx + num_records_per_thread;
        
	if(thread_end_idx> pgc.reduced_key_vals.d_reduced_keyval_arr_len)
                thread_end_idx = pgc.reduced_key_vals.d_reduced_keyval_arr_len;
        if (thread_start_idx >= thread_end_idx)
                return;

        int val_pos=0, key_pos=0;
        for (int i=0; i<thread_start_idx; i++){
                val_pos += (pgc.reduced_key_vals.d_reduced_keyval_arr[i].valSize+3)/4*4;
                key_pos += (pgc.reduced_key_vals.d_reduced_keyval_arr[i].keySize+3)/4*4;
        }//for
	
        for (int i = thread_start_idx; i < thread_end_idx;i++){
                memcpy( (char *)(pgc.output_key_vals.d_KeyBuff) + key_pos,
                        (char *)(pgc.reduced_key_vals.d_reduced_keyval_arr[i].key), pgc.reduced_key_vals.d_reduced_keyval_arr[i].keySize);
                //key_pos += pgc.reduced_key_vals.d_reduced_keyval_arr[i].keySize;
                memcpy( (char *)(pgc.output_key_vals.d_ValBuff) + val_pos,
                        (char *)(pgc.reduced_key_vals.d_reduced_keyval_arr[i].val), pgc.reduced_key_vals.d_reduced_keyval_arr[i].valSize);
       		printf("[copyDataFromDevice2Host4Reduce] key:[%s] val:[%d]\n",(char *)pgc.output_key_vals.d_KeyBuff+key_pos,
					*(int *)pgc.output_key_vals.d_ValBuff+val_pos);
        
		key_pos += pgc.reduced_key_vals.d_reduced_keyval_arr[i].keySize;
		val_pos += pgc.reduced_key_vals.d_reduced_keyval_arr[i].valSize;
  
	 }//for
	
}//__global__

void PandaEmitCPUMapOutput(void *key, void *val, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx){
	
	if(map_task_idx >= pcc->input_key_vals.num_input_record) {	ErrorLog("error ! map_task_idx >= d_g_state->num_input_record");		return;	}
	keyval_arr_t *kv_arr_p = &(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_task_idx]);

	char *buff = (char*)(kv_arr_p->shared_buff);
	
	if (!((*kv_arr_p->shared_buff_pos) + keySize + valSize < (*kv_arr_p->shared_buff_len) - sizeof(keyval_pos_t)*((*kv_arr_p->shared_arr_len)+1))){
		WarnLog("Warning! not enough memory at CPU task:%d *kv_arr_p->shared_arr_len:%d current buff_size:%d KB",
			map_task_idx,*kv_arr_p->shared_arr_len,(*kv_arr_p->shared_buff_len)/1024);

		char *new_buff = (char*)malloc(sizeof(char)*((*kv_arr_p->shared_buff_len)*2));
		if(new_buff==NULL){ ErrorLog("Error ! There is not enough memory to allocat!"); return; }

		memcpy(new_buff, buff, sizeof(char)*(*kv_arr_p->shared_buff_pos));
		int blockSize = sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len);
		memcpy(new_buff + (*kv_arr_p->shared_buff_len)*2 - blockSize, 
			(char*)buff + (*kv_arr_p->shared_buff_len) - blockSize,
														blockSize);
		
		(*kv_arr_p->shared_buff_len) = 2*(*kv_arr_p->shared_buff_len);
		for(int  idx = 0; idx < (kv_arr_p->shared_buddy_len); idx++){
			int cur_map_task_idx = kv_arr_p->shared_buddy[idx];  //the buddy relationship won't be changed
			keyval_arr_t *cur_kv_arr_p = &(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[cur_map_task_idx]);
			cur_kv_arr_p->shared_buff = new_buff;
		}//for
		free(buff);//
		buff = new_buff;
	}//if
	
	keyval_pos_t *kv_p = (keyval_pos_t *)((char *)buff + *kv_arr_p->shared_buff_len - sizeof(keyval_pos_t)*((*kv_arr_p->shared_arr_len)+1));
	(*kv_arr_p->shared_arr_len)++;
	kv_p->task_idx = map_task_idx;
	kv_p->next_idx = _MAP;
	
	kv_p->keyPos = (*kv_arr_p->shared_buff_pos);
	*kv_arr_p->shared_buff_pos += ((keySize+3)/4)*4;		//alignment 4 bytes for reading and writing
	memcpy((char *)(buff) + kv_p->keyPos, key, keySize);
	kv_p->keySize = keySize;
	
	kv_p->valPos = (*kv_arr_p->shared_buff_pos);
	*kv_arr_p->shared_buff_pos += ((valSize+3)/4)*4;
	//char *val_p = (char *)(buff) + kv_p->valPos;
	memcpy((char *)(buff) + kv_p->valPos, val, valSize);
	kv_p->valSize = valSize;
	(kv_arr_p->arr) = kv_p;

}//

void PandaEmitCPUReduceOutput (	void*		key,
				void*		val,
				int		keySize,
				int		valSize,
				panda_cpu_context *pcc, int reduce_task_idx){
			keyval_t *p = (keyval_t *)(&(pcc->reduced_key_vals.reduced_keyval_arr[reduce_task_idx]));
			p->keySize = keySize;
			p->key = malloc(keySize);
			memcpy(p->key,key,keySize);
			p->valSize = valSize;
			p->val = malloc(valSize);
			memcpy(p->val,val,valSize);
			ShowLog("reduce output key:%s  val:%d",(char*)key,*(int *)val);

}

__device__ void PandaEmitGPUMapOutput(void *key, void *val, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx){
	
	keyval_arr_t *kv_arr_p = pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p[map_task_idx];
	char *buff = (char*)(kv_arr_p->shared_buff);
	
	int shared_buff_len		= *kv_arr_p->shared_buff_len;
	int shared_arr_len		= *kv_arr_p->shared_arr_len;
	int shared_buff_pos		= *kv_arr_p->shared_buff_pos;
	int required_mem_len	= (shared_buff_pos) + keySize + valSize + sizeof(keyval_pos_t)*(shared_arr_len+1);
	//if (!((*kv_arr_p->shared_buff_pos) + keySize + valSize <    - sizeof(keyval_pos_t)*((*kv_arr_p->shared_arr_len)+1))){
	
	printf("emitGPUMapOUtput shared_buff_pos:%d keySize:%d valSize:%d required_mem_len:%d shared_buff_len:%d\n",shared_buff_pos,keySize,valSize,required_mem_len,shared_buff_len);

	if (required_mem_len > shared_buff_len){

		while (required_mem_len >= shared_buff_len){
			shared_buff_len *= 2;
		}//while
		
		WarnLog("Warning! not enough memory at GPU task:%d *kv_arr_p->shared_arr_len:%d current buff_size:%d KB",
			map_task_idx,*kv_arr_p->shared_arr_len,(*kv_arr_p->shared_buff_len)/1024);
		
		char *new_buff = (char*)malloc(sizeof(char)*(shared_buff_len));
		if(new_buff==NULL){ WarnLog("Error ! There is not enough memory to allocat!"); return; }
		
		memcpy(new_buff, buff, sizeof(char)*(*kv_arr_p->shared_buff_pos));
		memcpy(new_buff + (shared_buff_len) - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len), 
			(char*)buff + (*kv_arr_p->shared_buff_len) - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len),
														sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len));
				
		(*kv_arr_p->shared_buff_len) = (shared_buff_len);
				
		for(int  idx = 0; idx < (kv_arr_p->shared_buddy_len); idx++){
				
			int cur_map_task_idx = kv_arr_p->shared_buddy[idx];  //the buddy relationship won't be changed 
			keyval_arr_t *cur_kv_arr_p = pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p[cur_map_task_idx];
			cur_kv_arr_p->shared_buff = new_buff;
				
		}//for
		free(buff);//?????
		buff = new_buff;
	}//if
	
	keyval_pos_t *kv_p = (keyval_pos_t *)((char *)buff + *kv_arr_p->shared_buff_len - sizeof(keyval_pos_t)*((*kv_arr_p->shared_arr_len)+1));
	(*kv_arr_p->shared_arr_len)++;
	kv_p->task_idx = map_task_idx;
	kv_p->next_idx = _MAP;

	kv_p->keyPos = (*kv_arr_p->shared_buff_pos);
	*kv_arr_p->shared_buff_pos += ((keySize+3)/4)*4;		//alignment 4 bytes for reading and writing
	memcpy((char *)(buff) + kv_p->keyPos,key,keySize);
	kv_p->keySize = keySize;

	//printf("emitGPUMapOutput key:%s\n",buff);

	kv_p->valPos = (*kv_arr_p->shared_buff_pos);
	*kv_arr_p->shared_buff_pos += ((valSize+3)/4)*4;
	//char *val_p = (char *)(buff) + kv_p->valPos;
	memcpy((char *)(buff) + kv_p->valPos, val, valSize);
	kv_p->valSize = valSize;
	(kv_arr_p->arr) = kv_p;
	//printf("emitGPUMapOutput key:%s val:%d kv_arr_p->arr_len:%d\n",(char *)(buff+kv_p->keyPos),*((int *)(buff+kv_p->valPos)),kv_arr_p->arr_len);
	kv_arr_p->arr_len++;
	printf("emitGPUMapOutput key:%s val:%d kv_arr_p->arr_len:%d map_task_idx:%d\n",(char *)(buff+kv_p->keyPos),*((int *)(buff+kv_p->valPos)),kv_arr_p->arr_len,map_task_idx);
	pgc->intermediate_key_vals.d_intermediate_keyval_total_count[map_task_idx] = kv_arr_p->arr_len;

}//__device__

__device__ void PandaEmitGPUCombinerOutput(void *key, void *val, int keySize, int valSize, panda_gpu_context *pgc, int map_task_idx){
			
	keyval_arr_t *kv_arr_p	= pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p[map_task_idx];
	void *shared_buff		= kv_arr_p->shared_buff;
	int shared_buff_len		= *kv_arr_p->shared_buff_len;
	int shared_arr_len		= *kv_arr_p->shared_arr_len;
	int shared_buff_pos		= *kv_arr_p->shared_buff_pos;
		
	int required_mem_len = (shared_buff_pos) + keySize + valSize + sizeof(keyval_pos_t)*(shared_arr_len+1);
	if (required_mem_len> shared_buff_len){

		while (required_mem_len>= shared_buff_len){
			shared_buff_len *= 2;
		}//while

		WarnLog("Warning! no enough memory in GPU task:%d need:%d KB KeySize:%d ValSize:%d shared_arr_len:%d shared_buff_pos:%d shared_buff_len:%d",
			map_task_idx, required_mem_len/1024,keySize,valSize,shared_arr_len,shared_buff_pos,shared_buff_len);
		
		char *new_buff = (char*)malloc(sizeof(char)*(shared_buff_len));
		if(new_buff==NULL)WarnLog(" There is not enough memory to allocat!");

		memcpy(new_buff, shared_buff, sizeof(char)*(*kv_arr_p->shared_buff_pos));
		memcpy(new_buff + (shared_buff_len) - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len), 
			(char*)shared_buff + (*kv_arr_p->shared_buff_len) - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len),
												sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len));
		
		(*kv_arr_p->shared_buff_len) = shared_buff_len;	
		
		for(int  idx = 0; idx < (kv_arr_p->shared_buddy_len); idx++){

		int cur_map_task_idx = kv_arr_p->shared_buddy[idx];  //the buddy relationship won't be changed 
		keyval_arr_t *cur_kv_arr_p = pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p[cur_map_task_idx];
		cur_kv_arr_p->shared_buff = new_buff;
		
		}//for

		free(shared_buff);
		shared_buff = new_buff;
	
	}//if

	keyval_pos_t *kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len + 1));
	kv_p->keySize = keySize;
	kv_p->valSize = valSize;
	kv_p->task_idx = map_task_idx;
	kv_p->next_idx = _COMBINE;			//merged results

	memcpy( (char*)shared_buff + *kv_arr_p->shared_buff_pos, key, keySize);
	kv_p->keyPos = *kv_arr_p->shared_buff_pos;
	*kv_arr_p->shared_buff_pos += (keySize+3)/4*4;

	memcpy( (char*)shared_buff + *kv_arr_p->shared_buff_pos, val, valSize);
	kv_p->valPos = *kv_arr_p->shared_buff_pos;
	*kv_arr_p->shared_buff_pos += (valSize+3)/4*4;
	
	(*kv_arr_p->shared_arr_len)++;
			
}//__device__

__device__ void PandaEmitGPUReduceOutput(
						void*		key, 
						void*		val, 
						int		keySize, 
						int		valSize,
						panda_gpu_context *pgc){
	//printf("[PandaGPUEmitReduceOutput] key:[%s] val:[%d] TID:%d len:%d\n",(char *)key,*(int *)val,TID,
	//pgc->reduced_key_vals.d_reduced_keyval_arr_len);
			
		        keyval_t *p = &(pgc->reduced_key_vals.d_reduced_keyval_arr[TID]);
			p->keySize = keySize;
			p->key = malloc(keySize);
			memcpy(p->key,key,keySize);
			p->valSize = valSize;
			p->val = malloc(valSize);
			memcpy(p->val,val,valSize);

}//__device__ 


void PandaEmitCPUCombinerOutput(void *key, void *val, int keySize, int valSize, panda_cpu_context *pcc, int map_task_idx){

	keyval_arr_t *kv_arr_p	= &(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_task_idx]);
	void *shared_buff		= kv_arr_p->shared_buff;
	int shared_buff_len		= *kv_arr_p->shared_buff_len;
	int shared_arr_len		= *kv_arr_p->shared_arr_len;
	int shared_buff_pos		= *kv_arr_p->shared_buff_pos;
	ShowLog("EmitCPUCombinerOutput shared_buff_len:%d shared_arr_len:%d shared_buff_pos:%d",shared_buff_len,shared_arr_len,shared_buff_pos);
	int required_mem_len = (shared_buff_pos) + keySize + valSize + sizeof(keyval_pos_t)*(shared_arr_len+1);
	if (required_mem_len> shared_buff_len){

		while(required_mem_len> shared_buff_len){
			shared_buff_len *= 2;
		}//while

		WarnLog("Warning! no enough memory in GPU task:%d need:%d KB KeySize:%d ValSize:%d shared_arr_len:%d shared_buff_pos:%d shared_buff_len:%d",
			map_task_idx, required_mem_len/1024,keySize,valSize,shared_arr_len,shared_buff_pos,shared_buff_len);
		
		char *new_buff = (char*)malloc(sizeof(char)*(shared_buff_len));
		if(new_buff==NULL)ErrorLog(" There is not enough memory to allocat!");

		memcpy(new_buff, shared_buff, sizeof(char)*(*kv_arr_p->shared_buff_pos));
		memcpy(new_buff + (shared_buff_len) - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len), 
			(char*)shared_buff + (*kv_arr_p->shared_buff_len) - sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len),
												sizeof(keyval_pos_t)*(*kv_arr_p->shared_arr_len));
		
		(*kv_arr_p->shared_buff_len) = shared_buff_len;	
		
		for(int  idx = 0; idx < (kv_arr_p->shared_buddy_len); idx++){

		int cur_map_task_idx = kv_arr_p->shared_buddy[idx];			//the buddy relationship won't be changed 
		keyval_arr_t *cur_kv_arr_p = &(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[cur_map_task_idx]);
		cur_kv_arr_p->shared_buff = new_buff;
		
		}//for

		free(shared_buff);
		shared_buff = new_buff;
	
	}//if

	keyval_pos_t *kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(shared_arr_len + 1));
	kv_p->keySize = keySize;
	kv_p->valSize = valSize;
	kv_p->task_idx = map_task_idx;
	kv_p->next_idx = _COMBINE;				//merged results

	memcpy( (char*)shared_buff + *kv_arr_p->shared_buff_pos, key, keySize);
	kv_p->keyPos = *kv_arr_p->shared_buff_pos;
	*kv_arr_p->shared_buff_pos += (keySize+3)/4*4;

	memcpy( (char*)shared_buff + *kv_arr_p->shared_buff_pos, val, valSize);
	kv_p->valPos = *kv_arr_p->shared_buff_pos;
	*kv_arr_p->shared_buff_pos += (valSize+3)/4*4;
	
	(*kv_arr_p->shared_arr_len)++;

}//void

__global__ void RunPandaGPUCombiner(panda_gpu_context pgc)
{

	//GpuErrorLog("gridDim.x:%d gridDim.y:%d gridDim.z:%d blockDim.x:%d blockDim.y:%d blockDim.z:%d blockIdx.x:%d blockIdx.y:%d blockIdx.z:%d",
	// gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z,blockIdx.x,blockIdx.y,blockIdx.z);

	int num_records_per_thread = (pgc.input_key_vals.num_input_record + (gridDim.x*blockDim.x*blockDim.y)-1)/(gridDim.x*blockDim.x*blockDim.y);
	//GpuErrorLog("num_records_per_thread:%d",num_records_per_thread);
	
	int block_start_idx = num_records_per_thread * blockIdx.x * blockDim.x * blockDim.y;

	int thread_start_idx = block_start_idx 
		+ ((threadIdx.y*blockDim.x + threadIdx.x)/STRIDE)*num_records_per_thread*STRIDE
		+ ((threadIdx.y*blockDim.x + threadIdx.x)%STRIDE);

	int thread_end_idx = thread_start_idx + num_records_per_thread;//*STRIDE;
	if (thread_end_idx > pgc.input_key_vals.num_input_record)
		thread_end_idx = pgc.input_key_vals.num_input_record;
	
	if (thread_start_idx >= thread_end_idx)
		return;

	//printf("lihuix [RunPandaGPUCombiner] thread_start_idx:%d thread_end_idx:%d =========",thread_start_idx, thread_end_idx);

	keyval_arr_t *kv_arr_p = pgc.intermediate_key_vals.d_intermediate_keyval_arr_arr_p[thread_start_idx];
	//int *buddy = kv_arr_p->shared_buddy;

	int unmerged_shared_arr_len = *kv_arr_p->shared_arr_len;
	//GpuErrorLog("[GPUCombiner] unmerged_shared_arr_len:%d",unmerged_shared_arr_len);

	val_t *val_t_arr = (val_t *)malloc(sizeof(val_t)*unmerged_shared_arr_len);
	if (val_t_arr == NULL) {
	GpuErrorLog("[GPUCombiner] there is no enough memory. Return");
	return;
	}//if

	int num_keyval_pairs_after_combiner = 0;
	for (int i=0; i<unmerged_shared_arr_len;i++){
		
		char *shared_buff	= (kv_arr_p->shared_buff);	
		int shared_buff_len = *kv_arr_p->shared_buff_len;

		keyval_pos_t *head_kv_p = (keyval_pos_t *)(shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(unmerged_shared_arr_len-i));
		keyval_pos_t *first_kv_p = head_kv_p;

		if (first_kv_p->next_idx != _MAP)
			continue;

		int iKeySize = first_kv_p->keySize;
		char *iKey = shared_buff + first_kv_p->keyPos;
		//char *iVal = shared_buff + first_kv_p->valPos;

		if((first_kv_p->keyPos%4!=0)||(first_kv_p->valPos%4!=0)){
			GpuErrorLog("keyPos or valPos is not aligned with 4 bytes, results could be wrong");
			return;
		}//if

		int index = 0;
		first_kv_p = head_kv_p;

		(val_t_arr[index]).valSize = first_kv_p->valSize;
		(val_t_arr[index]).val = (char*)shared_buff + first_kv_p->valPos;

		for (int j=i+1;j<unmerged_shared_arr_len;j++){

			keyval_pos_t *next_kv_p = (keyval_pos_t *)((char *)shared_buff + shared_buff_len - sizeof(keyval_pos_t)*(unmerged_shared_arr_len-j));
			char *jKey = (char *)shared_buff+next_kv_p->keyPos;
			int jKeySize = next_kv_p->keySize;
		
			if (gpu_compare(iKey,iKeySize,jKey,jKeySize)!=0){
				continue;
			}//if
			index++;
			first_kv_p->next_idx = j;
			first_kv_p = next_kv_p;
			(val_t_arr[index]).valSize = next_kv_p->valSize;
			(val_t_arr[index]).val = (char*)shared_buff + next_kv_p->valPos;

		}//for

		int valCount = index+1;
		if(valCount>1)
			panda_gpu_core_combiner(iKey,val_t_arr,iKeySize,(valCount),&pgc,thread_start_idx);
		else{
			first_kv_p->next_idx = _COMBINE;
			first_kv_p->task_idx = thread_start_idx;
		}
		num_keyval_pairs_after_combiner++;
	}//for
	free(val_t_arr);
	pgc.intermediate_key_vals.d_intermediate_keyval_total_count[thread_start_idx] = num_keyval_pairs_after_combiner;
	__syncthreads();

}//GPU

}
//#endif //__PANDA_LIB_CU__
