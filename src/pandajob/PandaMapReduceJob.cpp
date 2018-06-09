
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*    Panda Code V0.42                                            11/04/2017 */
/*                                                       huili@ruijie.com.cn */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "Panda.h"

#include <panda/Chunk.h>
#include <panda/Mapper.h>
#include <panda/Reducer.h>
#include <panda/Combiner.h>
#include <panda/Partitioner.h>
#include <panda/PandaMessage.h>
#include <panda/PandaMapReduceJob.h>
#include <panda/EmitConfiguration.h>

#include <algorithm>
#include <vector>
#include <cstring>
#include <string>

#include <mpi.h>

int gCommRank = 0;

namespace panda
{

  void PandaMapReduceJob::AddGPUReduceTask(panda_gpu_context* pgc, panda_node_context *pnc, int start_task_id, int end_task_id)
  {
	
	if(end_task_id <= start_task_id)
		return;
	keyvals_t * sorted_intermediate_keyvals_arr = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr;
		
	int total_count = 0;
	for(int i=start_task_id;i<end_task_id;i++){
		total_count += sorted_intermediate_keyvals_arr[i].val_arr_len;
	}//for
	//ShowLog("[AddGPUReduceTask] start_task_id:%d end_task_id:%d  total_count:%d", start_task_id, end_task_id, total_count);
		
	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=start_task_id;i<end_task_id;i++){
		totalKeySize += (sorted_intermediate_keyvals_arr[i].keySize+3)/4*4;
		for (int j=0;j<sorted_intermediate_keyvals_arr[i].val_arr_len;j++)
		totalValSize += (sorted_intermediate_keyvals_arr[i].vals[j].valSize+3)/4*4;
	}//for
	
	ShowLog("[AddGPUReduceTask] start_task_id:%d end_task_id:%d totalKeySize:%d totalValSize:%d  total_count:%d", 
		start_task_id, end_task_id, totalKeySize, totalValSize, total_count);

	cudaMalloc((void **)&(pgc->sorted_key_vals.d_sorted_keys_shared_buff), sizeof(char)*totalKeySize);
	cudaMalloc((void **)&(pgc->sorted_key_vals.d_sorted_vals_shared_buff), sizeof(char)*totalValSize);
	cudaMalloc((void **)&(pgc->sorted_key_vals.d_keyval_pos_arr), sizeof(keyval_pos_t)*total_count);

	pgc->sorted_key_vals.h_sorted_keys_shared_buff = malloc(sizeof(char)*totalKeySize);
	pgc->sorted_key_vals.h_sorted_vals_shared_buff = malloc(sizeof(char)*totalValSize);
		
	char *sorted_keys_shared_buff = (char *)pgc->sorted_key_vals.h_sorted_keys_shared_buff;
	char *sorted_vals_shared_buff = (char *)pgc->sorted_key_vals.h_sorted_vals_shared_buff;
	char *keyval_pos_arr = (char *)malloc(sizeof(keyval_pos_t)*total_count);
		
	//ShowLog("start_task_id:%d end_task_id:%d  total_count:%d", start_task_id, end_task_id, total_count);
		
	int sorted_key_arr_len = end_task_id - start_task_id;
	keyval_pos_t *tmp_keyval_pos_arr = (keyval_pos_t *)malloc(sizeof(keyval_pos_t)*total_count);
		
	int *pos_arr_4_pos_arr = (int*)malloc(sizeof(int)*(sorted_key_arr_len));
	memset(pos_arr_4_pos_arr,0,sizeof(int)*sorted_key_arr_len);
	int index = 0;
	int keyPos = 0;
	int valPos = 0;
	for (int i = start_task_id; i < end_task_id; i++){
		keyvals_t* p = (keyvals_t*)&(sorted_intermediate_keyvals_arr[i]);
		memcpy(sorted_keys_shared_buff+keyPos,p->key, p->keySize);
		for (int j=0;j<p->val_arr_len;j++){
			tmp_keyval_pos_arr[index].keyPos = keyPos;
			tmp_keyval_pos_arr[index].keySize = p->keySize;
			tmp_keyval_pos_arr[index].valPos = valPos;
			tmp_keyval_pos_arr[index].valSize = p->vals[j].valSize;
			memcpy(sorted_vals_shared_buff + valPos,p->vals[j].val,p->vals[j].valSize);
			valPos += (p->vals[j].valSize+3)/4*4;
			index++;
		}//for
		keyPos += (p->keySize+3)/4*4;
		pos_arr_4_pos_arr[i-start_task_id] = index;
	}//
	pgc->sorted_key_vals.d_sorted_keyvals_arr_len = end_task_id-start_task_id;
	cudaMemcpy(pgc->sorted_key_vals.d_keyval_pos_arr,tmp_keyval_pos_arr,sizeof(keyval_pos_t)*total_count,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&(pgc->sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr),sizeof(int)*sorted_key_arr_len);
	cudaMemcpy(pgc->sorted_key_vals.d_pos_arr_4_sorted_keyval_pos_arr,pos_arr_4_pos_arr,sizeof(int)*sorted_key_arr_len,cudaMemcpyHostToDevice);
	cudaMemcpy(pgc->sorted_key_vals.d_sorted_keys_shared_buff, sorted_keys_shared_buff, sizeof(char)*totalKeySize,cudaMemcpyHostToDevice);
	cudaMemcpy(pgc->sorted_key_vals.d_sorted_vals_shared_buff, sorted_vals_shared_buff, sizeof(char)*totalValSize,cudaMemcpyHostToDevice);
  }

  void PandaMapReduceJob::AddCPUReduceTask(panda_cpu_context* pcc, panda_node_context *pnc, int start_task_id, int end_task_id)
  {
	if (end_task_id <= start_task_id)
	{
		ErrorLog("end_task_id<=start_task_id Warning!");
		pcc->sorted_key_vals.sorted_keyvals_arr_len = 0;
		return;
	}//if

	int len = pnc->sorted_key_vals.sorted_keyvals_arr_len;
	if (len < (end_task_id - start_task_id) )
	{
		ErrorLog("pnc->sorted_key_vals.sorted_keyvals_arr_len < end_task_id - start_task_id Warning!");
		pcc->sorted_key_vals.sorted_keyvals_arr_len = 0;
		return;
	}

	if (len == 0) {
		ErrorLog("pnc->sorted_key_vals.sorted_keyvals_arr_len <= 0 Warning!");
		pcc->sorted_key_vals.sorted_keyvals_arr_len = 0;
		return;
	}
	pcc->sorted_key_vals.sorted_intermediate_keyvals_arr = (keyvals_t *)malloc(sizeof(keyvals_t)*(end_task_id - start_task_id));
	pcc->sorted_key_vals.totalKeySize = pnc->sorted_key_vals.totalKeySize;
	pcc->sorted_key_vals.totalValSize = pnc->sorted_key_vals.totalValSize;
	
	for (int i = 0; i< end_task_id - start_task_id; i++){
	pcc->sorted_key_vals.sorted_intermediate_keyvals_arr[i].keySize		= pnc->sorted_key_vals.sorted_intermediate_keyvals_arr[start_task_id+i].keySize;
	pcc->sorted_key_vals.sorted_intermediate_keyvals_arr[i].key		= pnc->sorted_key_vals.sorted_intermediate_keyvals_arr[start_task_id+i].key;
	pcc->sorted_key_vals.sorted_intermediate_keyvals_arr[i].vals		= pnc->sorted_key_vals.sorted_intermediate_keyvals_arr[start_task_id+i].vals;
	pcc->sorted_key_vals.sorted_intermediate_keyvals_arr[i].val_arr_len = pnc->sorted_key_vals.sorted_intermediate_keyvals_arr[start_task_id+i].val_arr_len;
	//ShowLog("cpu_key:%s val_len:%d",
	//	pcc->sorted_key_vals.sorted_intermediate_keyvals_arr[i].key,pcc->sorted_key_vals.sorted_intermediate_keyvals_arr[i].val_arr_len);

	}//for
	pcc->sorted_key_vals.sorted_keyvals_arr_len = end_task_id - start_task_id;
  }//void

  void PandaMapReduceJob::StartPandaLocalMergeGPUOutput()
  {
	ExecutePandaGPUShuffleMerge(this->pNodeContext, this->pGPUContext);
  }//void

  void PandaMapReduceJob::StartPandaGPUSortTasks()
  {
	ExecutePandaGPUSort(this->pGPUContext);
  }//int

  void PandaMapReduceJob::StartPandaCPUSortTasks()
  {
	ExecutePandaCPUSort(this->pCPUContext, this->pNodeContext);
  }//int

  void PandaMapReduceJob::StartPandaGPUReduceTasks()
  {
	ExecutePandaGPUReduceTasks(this->pGPUContext);
  }// int PandaMapReduceJob


  void PandaMapReduceJob::StartPandaCPUMapTasks()
  {
	panda_cpu_context *pcc = this->pCPUContext;
	panda_node_context *pnc = this->pNodeContext;

	if (pcc->input_key_vals.num_input_record < 0)		{ ShowLog("Error: no any input keys");			exit(-1);}
	if (pcc->input_key_vals.input_keyval_arr == NULL)	{ ShowLog("Error: input_keyval_arr == NULL");		exit(-1);}
	if (pcc->num_cpus_cores <= 0)				{ ErrorLog("Error: pcc->num_cpus == 0");		exit(-1);}

	int num_cpus_cores = pcc->num_cpus_cores;
	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=0; i<pcc->input_key_vals.num_input_record; i++){
		totalKeySize += pcc->input_key_vals.input_keyval_arr[i].keySize;
		totalValSize += pcc->input_key_vals.input_keyval_arr[i].valSize;
	}//for

	ShowLog("num_input_record:%d, totalKeySize:%f KB totalValSize:%f KB num_cpus:%d", 
		pcc->input_key_vals.num_input_record, (float)(totalKeySize)/1024.0, (float)(totalValSize)/1024.0, pcc->num_cpus_cores);

	pcc->panda_cpu_task_thread = (pthread_t *)malloc(sizeof(pthread_t)*(num_cpus_cores));
	pcc->panda_cpu_task_thread_info = (panda_cpu_task_info_t *)malloc(sizeof(panda_cpu_task_info_t)*(num_cpus_cores));
	pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*pcc->input_key_vals.num_input_record);
	pcc->intermediate_key_vals.intermediate_keyval_arr_arr_len = pcc->input_key_vals.num_input_record;
	memset(pcc->intermediate_key_vals.intermediate_keyval_arr_arr_p, 0, sizeof(keyval_arr_t)*pcc->input_key_vals.num_input_record);
	for (int i=0; i < num_cpus_cores; i++){
			
		pcc->panda_cpu_task_thread_info[i].pcc = (panda_cpu_context  *)(this->pCPUContext);
		pcc->panda_cpu_task_thread_info[i].pnc = (panda_node_context *)(this->pNodeContext);
		pcc->panda_cpu_task_thread_info[i].num_cpus_cores = num_cpus_cores;
		pcc->panda_cpu_task_thread_info[i].start_task_idx = 0;
		pcc->panda_cpu_task_thread_info[i].end_task_idx = 0;
			
	}//for

	pcc->intermediate_key_vals.intermediate_keyval_total_count = (int *)malloc(pcc->input_key_vals.num_input_record*sizeof(int));
	memset(pcc->intermediate_key_vals.intermediate_keyval_total_count, 0, pcc->input_key_vals.num_input_record * sizeof(int));
	
	keyval_arr_t *d_keyval_arr_p;
	int *count = NULL;
	
	int num_threads				= pcc->num_cpus_cores;
	int num_input_record		= pcc->input_key_vals.num_input_record;
	int num_records_per_thread	= (num_input_record)/(num_threads);
	
	int start_task_idx = 0;
	int end_task_idx = 0;
		
	for (int tid = 0; tid < num_threads; tid++){
		end_task_idx = start_task_idx + num_records_per_thread;
		if (tid < (num_input_record % num_threads))
			end_task_idx++;
		pcc->panda_cpu_task_thread_info[tid].start_task_idx = start_task_idx;
		if (end_task_idx > num_input_record) 
			end_task_idx = num_input_record;
		pcc->panda_cpu_task_thread_info[tid].end_task_idx = end_task_idx;
			
		//if(end_task_idx > start_task_idx)
		if (pthread_create(&(pcc->panda_cpu_task_thread[tid]),NULL,RunPandaCPUMapThread,(void *)&(pcc->panda_cpu_task_thread_info[tid])) != 0) 
			ErrorLog("Thread creation failed Tid:%d!",tid);
		start_task_idx = end_task_idx;
	}//for
	
	for (int tid = 0;tid<num_threads;tid++){
		void *exitstat;
		if (pthread_join(pcc->panda_cpu_task_thread[tid],&exitstat)!=0)
			ErrorLog("joining failed tid:%d",tid);
	}//for
  }//int PandaMapReduceJob::StartPandaCPUMapTasks()


  void PandaMapReduceJob::StartPandaGPUMapTasks()
  {		

	panda_gpu_context *pgc = this->pGPUContext;
	//-------------------------------------------------------
	//0, Check status of pgc;
	//-------------------------------------------------------
			
	if (pgc->input_key_vals.num_input_record<0)			{ ShowLog("Error: no any input keys"); exit(-1);}
	if (pgc->input_key_vals.h_input_keyval_arr == NULL) { ShowLog("Error: h_input_keyval_arr == NULL"); exit(-1);}

	//if (pgc->input_key_vals.num_mappers<=0) {pgc->num_mappers = (NUM_BLOCKS)*(NUM_THREADS);}
	//if (pgc->input_key_vals.num_reducers<=0) {pgc->num_reducers = (NUM_BLOCKS)*(NUM_THREADS);}

	//-------------------------------------------------------
	//1, prepare buffer to store input data
	//-------------------------------------------------------

	keyval_arr_t *h_keyval_arr_arr = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*pgc->input_key_vals.num_input_record);
	keyval_arr_t *d_keyval_arr_arr;
	cudaMalloc((void**)&(d_keyval_arr_arr),pgc->input_key_vals.num_input_record*sizeof(keyval_arr_t));
	
	for (int i=0; i<pgc->input_key_vals.num_input_record;i++){
		h_keyval_arr_arr[i].arr = NULL;
		h_keyval_arr_arr[i].arr_len = 0;
	}//for

	keyval_arr_t **d_keyval_arr_arr_p;
	cudaMalloc((void***)&(d_keyval_arr_arr_p),pgc->input_key_vals.num_input_record*sizeof(keyval_arr_t*));
	pgc->intermediate_key_vals.d_intermediate_keyval_arr_arr_p = d_keyval_arr_arr_p;

	int *count = NULL;
	cudaMalloc((void**)(&count),pgc->input_key_vals.num_input_record*sizeof(int));
	pgc->intermediate_key_vals.d_intermediate_keyval_total_count = count;
	cudaMemset(pgc->intermediate_key_vals.d_intermediate_keyval_total_count,0,
					pgc->input_key_vals.num_input_record*sizeof(int));
	//----------------------------------------------
	//3, determine the number of threads to run
	//----------------------------------------------
	
	//--------------------------------------------------
	//4, start_task_id map
	//Note: DO *NOT* set large number of threads within block (512), which lead to too many invocation of malloc in the kernel. 
	//--------------------------------------------------

	int numGPUCores = pgc->num_gpus_cores;//getGPUCoresNum();

	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
    	dim3 grids(numBlocks, 1);
	int total_gpu_threads = (grids.x*grids.y*blocks.x*blocks.y);
	ShowLog("GridDim.X:%d GridDim.Y:%d BlockDim.X:%d BlockDim.Y:%d TotalGPUThreads:%d",grids.x,grids.y,blocks.x,blocks.y,total_gpu_threads);

	cudaDeviceSynchronize();
	ExecutePandaGPUMapTasks(*pgc,grids,blocks);

	int num_records_per_thread = (pgc->input_key_vals.num_input_record + (total_gpu_threads)-1)/(total_gpu_threads);
	int totalIter = num_records_per_thread;
	//ShowLog("num_records_per_thread:%d totalIter:%d",num_records_per_thread, totalIter);

	for (int iter = 0; iter< totalIter; iter++){
		ExecutePandaGPUMapTasksIterative(*pgc, totalIter-1-iter, totalIter, grids, blocks);
		cudaThreadSynchronize();
		size_t total_mem,avail_mem;
		cudaMemGetInfo( &avail_mem, &total_mem );
		//ShowLog("GPU_ID:[%d] RunGPUMapTasks take %f sec at iter [%d/%d] remain %d mb GPU mem processed",
		//	pgc->gpu_id, t4-t3,iter,totalIter, avail_mem/1024/1024);
	}//for
	//ShowLog("GPU_ID:[%d] Done %d Tasks",pgc->gpu_id,pgc->num_input_record);
  }//int 


  void PandaMapReduceJob::InitPandaCPUMapReduce()
  {
	this->pCPUContext					= CreatePandaCPUContext();
	this->pCPUContext->input_key_vals.num_input_record	= cpuMapTasks.size();
	this->pCPUContext->input_key_vals.input_keyval_arr	= (keyval_t *)malloc(cpuMapTasks.size()*sizeof(keyval_t));
	this->pCPUContext->num_cpus_cores			= getCPUCoresNum();
	this->pCPUContext->cpu_mem_size = getCPUMemSizeGb();
	this->pCPUContext->cpu_mem_bandwidth = getCPUMemBandwidthGb();
	this->pCPUContext->cpu_GHz = getCPUGHz();
	for (unsigned int i= 0;i<cpuMapTasks.size();i++){

		void *key = this->cpuMapTasks[i]->key;
		int keySize = this->cpuMapTasks[i]->keySize;
		void *val = this->cpuMapTasks[i]->val;
		int valSize = this->cpuMapTasks[i]->valSize;
		this->pCPUContext->input_key_vals.input_keyval_arr[i].key = key;
		this->pCPUContext->input_key_vals.input_keyval_arr[i].keySize = keySize;
		this->pCPUContext->input_key_vals.input_keyval_arr[i].val = val;
		this->pCPUContext->input_key_vals.input_keyval_arr[i].valSize = valSize;

	}//for

	panda_cpu_context* pcc = this->pCPUContext;

	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=0;i<pcc->input_key_vals.num_input_record;i++){
		totalKeySize += pcc->input_key_vals.input_keyval_arr[i].keySize;
		totalValSize += pcc->input_key_vals.input_keyval_arr[i].valSize;
	}//for

	//ShowLog("GPU_ID:[%d] copy %d input records from Host to GPU memory totalKeySize:%d KB totalValSize:%d KB",
	//	pgc->gpu_id, pgc->num_input_record, totalKeySize/1024, totalValSize/1024);
	
	void *input_vals_shared_buff = malloc(totalValSize);
	void *input_keys_shared_buff = malloc(totalKeySize);
	
	keyval_pos_t *input_keyval_pos_arr = 
				(keyval_pos_t *)malloc(sizeof(keyval_pos_t)*pcc->input_key_vals.num_input_record);
	
	int keyPos  = 0;
	int valPos  = 0;
	int keySize = 0;
	int valSize = 0;
	
	for(int i=0;i<pcc->input_key_vals.num_input_record;i++){
		keySize = pcc->input_key_vals.input_keyval_arr[i].keySize;
		valSize = pcc->input_key_vals.input_keyval_arr[i].valSize;
		
		memcpy((char *)input_keys_shared_buff + keyPos,(char *)(pcc->input_key_vals.input_keyval_arr[i].key), keySize);
		memcpy((char *)input_vals_shared_buff + valPos,(char *)(pcc->input_key_vals.input_keyval_arr[i].val), valSize);
		
		input_keyval_pos_arr[i].keySize = keySize;
		input_keyval_pos_arr[i].keyPos = keyPos;
		input_keyval_pos_arr[i].valPos = valPos;
		input_keyval_pos_arr[i].valSize = valSize;

		keyPos += keySize;	
		valPos += valSize;
	}//for
  }//void



  void PandaMapReduceJob::InitPandaGPUMapReduce()
  {

	this->pGPUContext = CreatePandaGPUContext();
	this->pGPUContext->num_gpus_cores = getGPUCoresNum();
	this->pGPUContext->gpu_mem_size = getGPUMemSize();
	this->pGPUContext->gpu_mem_bandwidth = getGPUMemBandwidth();
	this->pGPUContext->gpu_GHz = getGPUGHz();

	this->pGPUContext->input_key_vals.num_input_record = gpuMapTasks.size();//Ratio
	this->pGPUContext->input_key_vals.h_input_keyval_arr = 	(keyval_t *)malloc(gpuMapTasks.size()*sizeof(keyval_t));

	//TODO for the test purpose only
	for (unsigned int i= 0;i<gpuMapTasks.size();i++){
		void *key = this->gpuMapTasks[i]->key;
		int keySize = this->gpuMapTasks[i]->keySize;
		void *val = this->gpuMapTasks[i]->val;
		int valSize = this->gpuMapTasks[i]->valSize;
		this->pGPUContext->input_key_vals.h_input_keyval_arr[i].key = key;
		this->pGPUContext->input_key_vals.h_input_keyval_arr[i].keySize = keySize;
		this->pGPUContext->input_key_vals.h_input_keyval_arr[i].val = val;			//didn't use memory copy
		this->pGPUContext->input_key_vals.h_input_keyval_arr[i].valSize = valSize;
	}//for

	panda_gpu_context* pgc = this->pGPUContext;

	int totalKeySize = 0;
	int totalValSize = 0;
	for(int i=0;i<pgc->input_key_vals.num_input_record;i++){
		totalKeySize += pgc->input_key_vals.h_input_keyval_arr[i].keySize;
		totalValSize += pgc->input_key_vals.h_input_keyval_arr[i].valSize;
	}//for

	//ShowLog("GPU_ID:[%d] copy %d input records from Host to GPU memory totalKeySize:%d KB totalValSize:%d KB",
	//	pgc->gpu_id, pgc->num_input_record, totalKeySize/1024, totalValSize/1024);

	void *input_vals_shared_buff = malloc(totalValSize);
	void *input_keys_shared_buff = malloc(totalKeySize);
	keyval_pos_t *input_keyval_pos_arr = 
		(keyval_pos_t *)malloc(sizeof(keyval_pos_t)*pgc->input_key_vals.num_input_record);
	
	int keyPos = 0;
	int valPos = 0;
	int keySize = 0;
	int valSize = 0;
	
	for(int i=0;i<pgc->input_key_vals.num_input_record;i++){
		
		keySize = pgc->input_key_vals.h_input_keyval_arr[i].keySize;
		valSize = pgc->input_key_vals.h_input_keyval_arr[i].valSize;
		
		memcpy((char *)input_keys_shared_buff + keyPos,(char *)(pgc->input_key_vals.h_input_keyval_arr[i].key), keySize);
		memcpy((char *)input_vals_shared_buff + valPos,(char *)(pgc->input_key_vals.h_input_keyval_arr[i].val), valSize);
		
		input_keyval_pos_arr[i].keySize = keySize;
		input_keyval_pos_arr[i].keyPos = keyPos;
		input_keyval_pos_arr[i].valPos = valPos;
		input_keyval_pos_arr[i].valSize = valSize;

		keyPos += keySize;	
		valPos += valSize;

	}//for

	cudaMalloc((void **)&pgc->input_key_vals.d_input_keyval_pos_arr,sizeof(keyval_pos_t)*pgc->input_key_vals.num_input_record);
	cudaMalloc((void **)&pgc->input_key_vals.d_input_keys_shared_buff, totalKeySize);
	cudaMalloc((void **)&pgc->input_key_vals.d_input_vals_shared_buff, totalValSize);
	cudaMemcpy(pgc->input_key_vals.d_input_keyval_pos_arr, input_keyval_pos_arr,sizeof(keyval_pos_t)*pgc->input_key_vals.num_input_record ,cudaMemcpyHostToDevice);
	cudaMemcpy(pgc->input_key_vals.d_input_keys_shared_buff, input_keys_shared_buff,totalKeySize ,cudaMemcpyHostToDevice);
	cudaMemcpy(pgc->input_key_vals.d_input_vals_shared_buff, input_vals_shared_buff,totalValSize ,cudaMemcpyHostToDevice);
	//checkCudaErrors(cudaMemcpy(pgc->d_input_keyval_arr,h_buff,sizeof(keyval_t)*pgc->num_input_record,cudaMemcpyHostToDevice));
	cudaThreadSynchronize();				
	//pgc->iterative_support = true;	
	
  }//void

  void PandaMapReduceJob::InitPandaRuntime(){
	
	this->pNodeContext = new panda_node_context;
	if (this->pNodeContext == NULL) exit(-1);
	memset(this->pNodeContext, 0, sizeof(panda_node_context));

	//if(this->messager == NULL) exit(-1);
	
	this->pNodeContext->sorted_key_vals.sorted_keyvals_arr_len = 0;
	int max_len = 100;//configurable
	this->pNodeContext->sorted_key_vals.sorted_keyval_arr_max_len = max_len;
	this->pNodeContext->sorted_key_vals.sorted_intermediate_keyvals_arr = (keyvals_t *)malloc(sizeof(keyvals_t)*max_len);

	if (this->commRank == 0){
		ShowLog("commRank:%d, commSize:%d",this->commRank, this->commSize);
		this->pRuntimeContext = new panda_runtime_context;
	}//if
	else
		this->pRuntimeContext = NULL;
	if (messager    != NULL) {
		messager->MsgInit();
		messager->setPnc(this->pNodeContext);
		StartPandaMessageThread();
	}else{
		ErrorLog("messager == NULL");
		exit(-1);
	}
	MPI_Barrier(MPI_COMM_WORLD);
  }//void

  
  
  void PandaMapReduceJob::StartPandaMessageThread()
  {
    MessageThread = new oscpp::Thread(messager);
    MessageThread->start();
  }


  void PandaMapReduceJob::WaitPandaMessagerExit()
  {
    if (messager!=NULL) messager->MsgFinish();
	MessageThread->join();

    	delete MessageThread;
  }//void

  PandaMapReduceJob::PandaMapReduceJob(int argc,char **argv)
    : MapReduceJob(argc, argv)
  {
  }

  PandaMapReduceJob::~PandaMapReduceJob()
  {
  }//PandaMapReduceJob

  //Significant change is required in this place
  void PandaMapReduceJob::addInput(Chunk * chunk)
  {
	//chunks.push_back(chunk);
	if(!this->getEnableCPU()&&!this->getEnableGPU()){
	ShowLog("neither GPU nor CPU are enabled");
	return;
	}
	
	if(!this->getEnableCPU()&&this->getEnableGPU()){
	addGPUMapTasks(chunk);
        ShowLog("addGPUMapTasks");
	return;
	}

	if(this->getEnableCPU()&&!this->getEnableGPU()){
	addCPUMapTasks(chunk);
	ShowLog("addCPUMapTasks");
	return;
	}

	if(this->getEnableCPU()&&this->getEnableGPU()){
	static bool lastgpu = false;
	static bool lastcpu = false;
	if(!lastcpu){
		addCPUMapTasks(chunk);
		ShowLog("addCPUMapTasks");
		lastcpu = true;
		lastgpu = false;
		return;
	}
	if(!lastgpu){
		addGPUMapTasks(chunk);
		ShowLog("addGPUMapTasks");
		lastcpu = false;
		lastgpu = true;
		return;	
	}
	}//if
  }//void

  void PandaMapReduceJob::addCPUMapTasks(panda::Chunk *chunk)
  {
	  void *key	= chunk->getKey();
	  void *val	= chunk->getVal();
	  int keySize	= chunk->getKeySize();
	  int valSize	= chunk->getValSize();
	  MapTask *pMapTask = new MapTask(keySize,key,valSize,val);
	  cpuMapTasks.push_back(pMapTask);
  }//void

  void PandaMapReduceJob::addGPUMapTasks(panda::Chunk *chunk)
  {

	  void *key = chunk->getKey();
	  void *val = chunk->getVal();
	  int keySize = chunk->getKeySize();
	  int valSize = chunk->getValSize();

	  MapTask *pMapTask = new MapTask(keySize,key,valSize,val);
	  gpuMapTasks.push_back(pMapTask);

  }//void

  void PandaMapReduceJob::StartPandaCPUCombineTasks()
  {
	  ExecutePandaCPUCombiner(this->pCPUContext);
  }//void

  void PandaMapReduceJob::StartPandaGPUCombineTasks()
  {
	  ExecutePandaGPUCombiner(this->pGPUContext);
  }//void 
			
  void PandaMapReduceJob::StartPandaSortBucket()
  {
	  ExecutePandaSortBucket(this->pNodeContext);
  }//for

  void PandaMapReduceJob::StartPandaCopyRecvedBucketToGPU(int start_task_id, int end_task_id)
  {
	  AddGPUReduceTask(this->pGPUContext,this->pNodeContext, start_task_id, end_task_id);
  }//void

  void PandaMapReduceJob::StartPandaCopyRecvedBucketToCPU(int start_task_id, int end_task_id)
  {
	  AddCPUReduceTask(this->pCPUContext,this->pNodeContext,start_task_id,end_task_id);
  }//void

  void PandaMapReduceJob::PandaAddKeyValue2Bucket(int bucketId, const char*key, int keySize, const char*val, int valSize)
  {

	  char * keyBuff = (char *)(this->pNodeContext->buckets.savedKeysBuff.at(bucketId));
	  char * valBuff = (char *)(this->pNodeContext->buckets.savedValsBuff.at(bucketId));

	  int keyBuffSize = this->pNodeContext->buckets.keyBuffSize[bucketId];
	  int valBuffSize = this->pNodeContext->buckets.valBuffSize[bucketId];
	  int *counts = this->pNodeContext->buckets.counts[bucketId];
	  
	  int curlen	 = counts[0];
	  int maxlen	 = counts[1];

	  int keyBufflen = counts[2];
	  int valBufflen = counts[3];

	  int *keyPosArray = this->pNodeContext->buckets.keyPos[bucketId];
	  int *valPosArray = this->pNodeContext->buckets.valPos[bucketId];
	  int *keySizeArray = this->pNodeContext->buckets.keySize[bucketId];
	  int *valSizeArray = this->pNodeContext->buckets.valSize[bucketId];

	  if (keyBufflen + keySize >= keyBuffSize){

			while(keyBufflen + keySize >= keyBuffSize)
				keyBuffSize *= 2;

			char *newKeyBuff = (char*)malloc(keyBuffSize);

			memcpy(newKeyBuff, keyBuff, keyBufflen);
			memcpy(newKeyBuff+keyBufflen, key, keySize);
			counts[2] = keyBufflen + keySize;
			this->pNodeContext->buckets.savedKeysBuff[bucketId] = newKeyBuff;
			this->pNodeContext->buckets.keyBuffSize[bucketId]   = keyBuffSize;
			//TODO remove keyBuff in std::vector
			delete [] keyBuff;
	  }else{
			memcpy(keyBuff + keyBufflen, key, keySize);
			counts[2] = keyBufflen+keySize;
	  }//else
	  
	  if (valBufflen + valSize >= valBuffSize){

		    while(valBufflen + valSize >= valBuffSize)
				valBuffSize *= 2;

		        char *newValBuff = (char*)malloc(valBuffSize);
			memcpy(newValBuff, valBuff, valBufflen);
			memcpy(newValBuff + valBufflen, val, valSize);
			counts[3] = valBufflen+valSize;
			this->pNodeContext->buckets.savedValsBuff[bucketId] = newValBuff;
			this->pNodeContext->buckets.valBuffSize[bucketId]   = valBuffSize;
			//TODO remove valBuff in std::vector
			delete [] valBuff;
	  }else{
			memcpy(valBuff + valBufflen, val, valSize);	//
			counts[3] = valBufflen+valSize;				//
	  }//else

	  keyPosArray[curlen]  = keyBufflen;
      	  valPosArray[curlen]  = valBufflen;
	  keySizeArray[curlen] = keySize;
	  valSizeArray[curlen] = valSize;

	  (counts[0])++;//increase one keyVal pair
	  if(counts[0] >= counts[1]){
		 
		 counts[1] *= 2;
		 int * newKeyPosArray = (int *)malloc(sizeof(int)*counts[1]);
		 int * newValPosArray = (int *)malloc(sizeof(int)*counts[1]);
		 int * newKeySizeArray = (int *)malloc(sizeof(int)*counts[1]);
		 int * newValSizeArray = (int *)malloc(sizeof(int)*counts[1]);

		 memcpy(newKeyPosArray, keyPosArray, sizeof(int)*counts[0]);
		 memcpy(newValPosArray, valPosArray, sizeof(int)*counts[0]);
		 memcpy(newKeySizeArray, keySizeArray, sizeof(int)*counts[0]);
		 memcpy(newValSizeArray, valSizeArray, sizeof(int)*counts[0]);

		 this->pNodeContext->buckets.keyPos[bucketId]  = newKeyPosArray;
		 this->pNodeContext->buckets.valPos[bucketId]  = newValPosArray;
		 this->pNodeContext->buckets.keySize[bucketId] = newKeySizeArray;
		 this->pNodeContext->buckets.valSize[bucketId] = newValSizeArray;
		 delete [] keyPosArray;
		 delete [] valPosArray;
	   	 delete [] keySizeArray;
 		 delete [] valSizeArray;
	  }//if
  }//void


  // send data to local bucket
  void PandaMapReduceJob::StartPandaDoPartitionOnCPU(){

	  //Init Buckets there are commSize buckets across cluster
	  //it is fixed, but can be extended in case there is not enough DRAM space
	  //TODO need to be configured in the future.
	  int keyBuffSize = 1024;
	  int valBuffSize = 1024;
	  int maxlen	  = 20;

	  this->pNodeContext->buckets.numBuckets  = this->commSize;
	  this->pNodeContext->buckets.keyBuffSize = new int[this->commSize];
	  this->pNodeContext->buckets.valBuffSize = new int[this->commSize];

	  for (int i=0; i<this->commSize; i++){
		  
		  this->pNodeContext->buckets.keyBuffSize[i] = keyBuffSize;
		  this->pNodeContext->buckets.valBuffSize[i] = valBuffSize;

		  char *keyBuff = new char[keyBuffSize];
		  char *valBuff = new char[valBuffSize];
		  this->pNodeContext->buckets.savedKeysBuff.push_back((char*)keyBuff);
		  this->pNodeContext->buckets.savedValsBuff.push_back((char*)valBuff);
		  int *keyPos  = new int[maxlen];
		  int *valPos  = new int[maxlen];
		  int *keySize = new int[maxlen];
		  int *valSize = new int[maxlen];
		  this->pNodeContext->buckets.keyPos.push_back(keyPos);
		  this->pNodeContext->buckets.valPos.push_back(valPos);
		  this->pNodeContext->buckets.valSize.push_back(valSize);
		  this->pNodeContext->buckets.keySize.push_back(keySize);
		  int* counts_i		= new int[4];
		  counts_i[0]		= 0;	//curlen
		  counts_i[1]		= maxlen;	
		  counts_i[2]		= 0;	//keybufflen
		  counts_i[3]		= 0;	//valbufflen
		  this->pNodeContext->buckets.counts.push_back(counts_i);

	  }//for

	  keyvals_t *sorted_intermediate_keyvals_arr1 = this->pNodeContext->sorted_key_vals.sorted_intermediate_keyvals_arr;
	  ShowLog("this->pNodeContext->sorted_key_vals.sorted_keyvals_arr_len:%d", this->pNodeContext->sorted_key_vals.sorted_keyvals_arr_len);

	  for (int i=0; i<this->pNodeContext->sorted_key_vals.sorted_keyvals_arr_len; i++){
		char *key	 = (char *)(sorted_intermediate_keyvals_arr1[i].key);
		int keySize  = sorted_intermediate_keyvals_arr1[i].keySize;
		int bucketId = partition->GetHash(key,keySize,this->commSize);
		val_t *vals  = sorted_intermediate_keyvals_arr1[i].vals;
		int len = sorted_intermediate_keyvals_arr1[i].val_arr_len;
		for (int j=0;j<len;j++){
			ShowLog("key:%s keySize:%d val:%d dump to buckeId:[%d]",key, keySize, *(int *)vals[j].val, bucketId);
			PandaAddKeyValue2Bucket(bucketId, (char *)key, keySize,(char *)(vals[j].val),vals[j].valSize);
		}//for
	  }//for
	  ShowLog("Panda adding Key/Values to bucket is done on local host.\n");
  }//void

  void PandaMapReduceJob::StartPandaCPUReduceTasks(){
	  ExecutePandaCPUReduceTasks(this->pCPUContext);
  }//void
  void PandaMapReduceJob::StartPandaCPUDumpReduceTasks(){
	  ExecutePandaCPUDumpReduceTasks(this->pCPUContext);
  }//void
  void PandaMapReduceJob::StartPandaPartitionSendData()
  {
    for (int index = 0; index < commSize; index++)
    {
	  int curlen	= this->pNodeContext->buckets.counts[index][0];
	  int maxlen	= this->pNodeContext->buckets.counts[index][1];
	  int keySize	= this->pNodeContext->buckets.counts[index][2];
	  int valSize	= this->pNodeContext->buckets.counts[index][3];

	  ShowLog("bucket index:%d curlen:%d maxlen:%d keySize:%d valSize:%d ",index,curlen,maxlen,keySize,valSize);

	  if(curlen == 0){
	  int dest = index;
	  keySize = -1;
	  valSize = -1;
          curlen  = -1;
	  oscpp::AsyncIORequest * ioReq = messager->sendTo(dest,
                                             		NULL,
                                             		NULL,
      							NULL,
							keySize,
							valSize,
							curlen);
	  //sendReqs.push_back(ioReq);
	  continue;
  	  }//if

	  char *keyBuff = this->pNodeContext->buckets.savedKeysBuff[index];
	  char *valBuff = this->pNodeContext->buckets.savedValsBuff[index];

	  int *keySizeArray = this->pNodeContext->buckets.keySize[index];
	  int *valSizeArray = this->pNodeContext->buckets.valSize[index];
	  int *keyPosArray  = this->pNodeContext->buckets.keyPos[index];
	  int *valPosArray  = this->pNodeContext->buckets.valPos[index];
	  
	  int *keyPosKeySizeValPosValSize = new int[curlen*4];
	  
	  if(keyPosArray==NULL)  ShowLog("Error");
	  if(valSizeArray==NULL) ShowLog("Error");
	  if(keyPosArray==NULL)  ShowLog("Error");
	  if(valPosArray==NULL)  ShowLog("Error");
 
	  for (int i=0; i<curlen; i++){
		keyPosKeySizeValPosValSize[i] = keyPosArray[i];
	  }//for
	  for (int i=curlen; i<curlen*2; i++){
		keyPosKeySizeValPosValSize[i] = keySizeArray[i-curlen];
	  }//for
	  for (int i=curlen*2; i<curlen*3; i++){
		keyPosKeySizeValPosValSize[i] = valPosArray[i-2*curlen];
	  }//for
	  for (int i=curlen*3; i<curlen*4; i++){
		keyPosKeySizeValPosValSize[i] = valSizeArray[i-3*curlen];
	  }//for

          //int dest = (index + commRank + commSize - 1) % commSize;
	  int dest = index;
          if (keySize+valSize > 0) // it can happen that we send nothing.
          {
          oscpp::AsyncIORequest * ioReq = messager->sendTo(dest,
                                                       	keyBuff,
                                                       	valBuff,
							keyPosKeySizeValPosValSize,
                                                       	keySize,
                                                     	valSize,
							curlen);
          //sendReqs.push_back(ioReq);
          }//if
    }//for
  }//void

  void PandaMapReduceJob::StartPandaGlobalPartition()
  {
	  StartPandaDoPartitionOnCPU();
	  StartPandaPartitionSendData();
          //if (syncPartSends) PandaPartitionCheckSends(true);
	  /*
	  this->pNodeContext->buckets.savedKeysBuff.clear();
	  this->pNodeContext->buckets.savedValsBuff.clear();
	  this->pNodeContext->buckets.keyPos.clear();
	  this->pNodeContext->buckets.valPos.clear();
	  this->pNodeContext->buckets.counts.clear();
	  */
  }//void

  void PandaMapReduceJob::execute()
  {

	InitPandaRuntime();
	if(this->getEnableGPU())
		InitPandaGPUMapReduce();
	if(this->getEnableCPU()){
		InitPandaCPUMapReduce();
	}//if
	if(this->getEnableGPU())
		StartPandaGPUMapTasks();
	if(this->getEnableCPU())
		StartPandaCPUMapTasks();

	if(this->getEnableGPU())
		StartPandaGPUCombineTasks();
	if(this->getEnableCPU())
		StartPandaCPUCombineTasks();

	if(this->getEnableGPU()){
		StartPandaGPUSortTasks();		
		StartPandaLocalMergeGPUOutput();	
	}//if
	if(this->getEnableCPU()){
		StartPandaCPUSortTasks();
	}//if

	StartPandaGlobalPartition();
	WaitPandaMessagerExit();
	//MPI_Barrier(MPI_COMM_WORLD);

	if(this->getEnableGPU()){
		this->pGPUContext->sorted_key_vals.d_sorted_keyvals_arr_len = 0;
	}
	if(this->getEnableCPU()){
		this->pCPUContext->sorted_key_vals.sorted_keyvals_arr_len = 0;
	}
	this->pNodeContext->sorted_key_vals.sorted_intermediate_keyvals_arr = NULL;
	this->pNodeContext->sorted_key_vals.sorted_keyvals_arr_len = 0;
	
	StartPandaSortBucket();

	int start_task_id = 0;
	int end_task_id = this->pNodeContext->sorted_key_vals.sorted_keyvals_arr_len;

	if(end_task_id>0){

	if(this->getEnableGPU()&&this->getEnableCPU()){
	StartPandaCopyRecvedBucketToCPU(start_task_id, end_task_id);
	//StartPandaCopyRecvedBucketToCPU(end_task_id/2+1, end_task_id);
	}//if
	if(this->getEnableGPU()&&(!this->getEnableCPU())){
	StartPandaCopyRecvedBucketToGPU(start_task_id, end_task_id);
	}
	if((!this->getEnableGPU())&&(!this->getEnableCPU())){
	return;
	}
	if((!this->getEnableGPU())&&this->getEnableCPU()){
	StartPandaCopyRecvedBucketToCPU(start_task_id, end_task_id);
	}

	}

	if(this->getEnableGPU())
		StartPandaGPUReduceTasks();
	if(this->getEnableCPU())
		StartPandaCPUReduceTasks();
	if(this->getEnableCPU())
		StartPandaCPUDumpReduceTasks();
    	//MPI_Barrier(MPI_COMM_WORLD);
	
  }
}
