#include <mpi.h>

#include <panda/Chunk.h>
#include <panda/Sorter.h>
#include <panda/Mapper.h>
#include <panda/Reducer.h>
#include <panda/Combiner.h>
#include <panda/Partitioner.h>
#include <panda/PandaMessage.h>
#include <panda/PartialReducer.h>

#include <panda/PandaMapReduceWorker.h>

#include <panda/PandaMapReduceJob.h>
#include <panda/EmitConfiguration.h>

#include <cudacpp/DeviceProperties.h>
#include <cudacpp/Runtime.h>
#include <cudacpp/Event.h>

#include "Panda.h"
#include <algorithm>
#include <vector>
#include <cstring>
#include <string>

extern int gCommRank;

namespace panda
{

  void PandaMapReduceJob::AddReduceTask4GPU(panda_gpu_context* pgc, panda_node_context *pnc, int start_row_id, int end_row_id){
	}
  void PandaMapReduceJob::AddReduceTask4CPU(panda_cpu_context* pgc, panda_node_context *pnc, int start_row_id, int end_row_id){
	}
  void PandaMapReduceJob::allocateReduceVariables(){
	}
  void PandaMapReduceJob::allocateMapVariables(){
	}
  void PandaMapReduceJob::determineMaximumSpaceRequirements(){
	}
  void PandaMapReduceJob::freeMapVariables(){
	}
  void PandaMapReduceJob::combine(){
	}
  void PandaMapReduceJob::mapChunkMemcpy(const unsigned int chunkIndex,
                                  	const void * const gpuKeySpace,
					const void * const gpuValueSpace){
	}

  void PandaMapReduceJob::mapChunkExecute(const unsigned int chunkIndex,
                                   	PandaGPUConfig & config,
					void * const memPool){
	}
  void PandaMapReduceJob::mapChunkPartition(const unsigned int chunkIndex,
                                     		void * const memPool,
						PandaGPUConfig & config){
	}

  void PandaMapReduceJob::queueChunk(const unsigned int chunkIndex){
	}
  void PandaMapReduceJob::partitionSubDoNullPartitioner(const int numKeys){
	} 
  void PandaMapReduceJob::partitionSubDoGPU(void * const memPool,
                                     void * const keySpace,
                                     void * const valueSpace,
                                     const int numKeys,
                                     const int singleKeySize,
				     const int singleValSize){
	}
  void PandaMapReduceJob::partitionSubSendData(const int singleKeySize, const int singleValSize){
	}
  void PandaMapReduceJob::saveChunk(const unsigned int chunkIndex){
	}
  void PandaMapReduceJob::globalPartition(){
	}
  void PandaMapReduceJob::collectVariablesFromMessageAndKill(){
	}
  void  PandaMapReduceJob::enqueueAllChunks(){
	}
  PandaGPUConfig & PandaMapReduceJob::getReduceConfig(const int index){
	}
  void PandaMapReduceJob::freeReduceVariables(){
	}
  void PandaMapReduceJob::copyReduceInput(int, int){
	}
  void PandaMapReduceJob::getReduceRunParameters(){
	}
  void PandaMapReduceJob::executeReduce(const int index){
	}
  void PandaMapReduceJob::collectTimings(){
	}
  void PandaMapReduceJob::copyReduceOutput(const int index, const int keysSoFar){
	}
  void PandaMapReduceJob::enqueueReductions(){
	}
  void PandaMapReduceJob::StartPandaAddReduceTask4GPU(int start_row_id, int end_row_id){
	}
  int PandaMapReduceJob::StartPandaGPUReduceTasks(){
	return 0;
	}
  void PandaMapReduceJob::addMapTasks(Chunk *chunk){
  	}  			
  void PandaMapReduceJob::StartPandaLocalMergeGPUOutput()
  {
	  ExecutePandaShuffleMergeGPU(this->pNodeContext, this->pGPUContext);
  }//void

  void PandaMapReduceJob::StartPandaSortGPUResults()
  {
	   ExecutePandaGPUSort(this->pGPUContext);
  }//int

  void PandaMapReduceJob::StartPandaSortCPUResults()
  {
	  ExecutePandaCPUSort(this->pCPUContext, this->pNodeContext);
  }//int

  void PandaMapReduceJob::StartPandaSortGPUCardResults(int a, int b)
  {
	  ExecutePandaGPUCardSort(this->pGPUCardContext, this->pNodeContext);
  }

  int PandaMapReduceJob::StartPandaReduceTasksOnGPUCard()
	{
		int gpu_id;
		cudaGetDevice(&gpu_id);
		ShowLog("Start Reduce Tasks on GPU:%d",gpu_id);
		ExecutePandaReduceTasksOnGPUCard(this->pGPUCardContext);
		return 0;
	}// int PandaMapReduceJob

 int PandaMapReduceJob::StartPandaReduceTasksOnGPU()
	{

		//InitGPUDevice(thread_info);
		/*panda_context *panda = (panda_context *)(thread_info->panda);
		gpu_context *d_g_state = (gpu_context *)(thread_info->d_g_state);
		int num_gpu_core_groups = d_g_state->num_gpu_core_groups;
		if ( num_gpu_core_groups <= 0){
			ShowError("num_gpu_core_groups == 0 return");
			return NULL;
		}*///if

		//TODO add input record gpu
		//AddReduceInputRecordGPU(d_g_state,(panda->sorted_intermediate_keyvals_arr), thread_info->start_idx, thread_info->end_idx);

		int gpu_id;
		cudaGetDevice(&gpu_id);
		ShowLog("Start GPU Reduce Tasks.  GPU_ID:%d",gpu_id);
		//ExecutePandaReduceTasksOnGPU(this->pGPUContext);
		return 0;
	}// int PandaMapReduceJob


	int PandaMapReduceJob::StartPandaCPUMapTasks()
	{
		panda_cpu_context *pcc = this->pCPUContext;
		panda_node_context *pnc = this->pNodeContext;

		if (pcc->input_key_vals.num_input_record < 0)		{ ShowLog("Error: no any input keys");			exit(-1);}
		if (pcc->input_key_vals.input_keyval_arr == NULL)	{ ShowLog("Error: input_keyval_arr == NULL");	exit(-1);}
		if (pcc->num_cpus_cores <= 0)						{ ShowError("Error: pcc->num_cpus == 0");		exit(-1);}

		
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
			pcc->panda_cpu_task_thread_info[i].start_row_idx = 0;
			pcc->panda_cpu_task_thread_info[i].end_row_idx = 0;
			
		}//for

		pcc->intermediate_key_vals.intermediate_keyval_total_count = (int *)malloc(pcc->input_key_vals.num_input_record*sizeof(int));
		memset(pcc->intermediate_key_vals.intermediate_keyval_total_count, 0, pcc->input_key_vals.num_input_record * sizeof(int));
	
		keyval_arr_t *d_keyval_arr_p;
		int *count = NULL;
	
		int num_threads				= pcc->num_cpus_cores;
		int num_input_record		= pcc->input_key_vals.num_input_record;
		int num_records_per_thread	= (num_input_record)/(num_threads);
	
		int start_row_idx = 0;
		int end_row_idx = 0;
		
		for (int tid = 0; tid < num_threads; tid++){
	
			end_row_idx = start_row_idx + num_records_per_thread;
			if (tid < (num_input_record % num_threads))
				end_row_idx++;
			pcc->panda_cpu_task_thread_info[tid].start_row_idx = start_row_idx;
			if (end_row_idx > num_input_record) 
				end_row_idx = num_input_record;
			pcc->panda_cpu_task_thread_info[tid].end_row_idx = end_row_idx;
			
			//if(end_row_idx > start_row_idx)
			if (pthread_create(&(pcc->panda_cpu_task_thread[tid]),NULL,ExecutePandaCPUMapThread,(void *)&(pcc->panda_cpu_task_thread_info[tid])) != 0) 
				ShowError("Thread creation failed Tid:%d!",tid);

			start_row_idx = end_row_idx;

		}//for
	
		for (int tid = 0;tid<num_threads;tid++){
			void *exitstat;
			if (pthread_join(pcc->panda_cpu_task_thread[tid],&exitstat)!=0)
				ShowError("joining failed tid:%d",tid);
		}//for
	
	}//int PandaMapReduceJob::StartPandaCPUMapTasks()

int PandaMapReduceJob::StartPandaMapTasksOnGPUCard()
{

	panda_gpu_card_context *pgcc = this->pGPUCardContext;

	if (pgcc->input_key_vals.num_input_record <0)
	{
		ShowLog("Error: no any input keys");
		exit(-1);
	}//if

	if (pgcc->input_key_vals.input_keyval_arr == NULL)
	{
		ShowLog("Error: input_keyval_arr == NULL");
		exit(-1);
	}

	pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_p = (keyval_arr_t *)malloc(sizeof(keyval_arr_t)*pgcc->input_key_vals.num_input_record);
	pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_len = pgcc->input_key_vals.num_input_record;
	pgcc->intermediate_key_vals.intermediate_keyval_total_count = (int *)malloc(pgcc->input_key_vals.num_input_record*sizeof(int));
	memset(pgcc->intermediate_key_vals.intermediate_keyval_total_count, 0, pgcc->input_key_vals.num_input_record * sizeof(int));


	char *buff		=	(char *)malloc(sizeof(char)*GPU_SHARED_BUFF_SIZE);
	int *int_arr	=	(int *)malloc(sizeof(int)*(pgcc->input_key_vals.num_input_record + 3));
	int *buddy		=	int_arr+3;
	
	int buddy_len	=	pgcc->input_key_vals.num_input_record;
	for (int i=0;i<buddy_len;i++){
		buddy [i]	=	i;
	}//for
	
	for (int map_idx = 0; map_idx < pgcc->input_key_vals.num_input_record; map_idx++){

		(pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff)		= buff;
		(pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_len)	= int_arr;
		(pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_pos)	= int_arr+1;
		(pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_arr_len)		= int_arr+2;
		
		*(pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_len)	= GPU_SHARED_BUFF_SIZE;
		*(pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buff_pos)	= 0;
		*(pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_arr_len)	= 0;
		(pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buddy)		= buddy;
		(pgcc->intermediate_key_vals.intermediate_keyval_arr_arr_p[map_idx].shared_buddy_len)	= buddy_len;

	}//for

	RunMapTasksOnGPUCardHost(*pgcc);

}


 //void InitGPUCardMapReduce(gpu_card_context* d_g_state)
 int PandaMapReduceJob::StartPandaGPUMapTasks()
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
	//1, prepare buffer to store intermediate results
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

	cudaThreadSynchronize();
	
	int numGPUCores = getGPUCoresNum();
	dim3 blocks(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	int numBlocks = (numGPUCores*16+(blocks.x*blocks.y)-1)/(blocks.x*blocks.y);
    	dim3 grids(numBlocks, 1);
	int total_gpu_threads = (grids.x*grids.y*blocks.x*blocks.y);
	ShowLog("GridDim.X:%d GridDim.Y:%d BlockDim.X:%d BlockDim.Y:%d TotalGPUThreads:%d",grids.x,grids.y,blocks.x,blocks.y,total_gpu_threads);

	cudaDeviceSynchronize();
	double t1 = PandaTimer();
	
	StartPandaGPUMapPartitioner(*pgc,grids,blocks);
	
	cudaThreadSynchronize();
	double t2 = PandaTimer();

	int num_records_per_thread = (pgc->input_key_vals.num_input_record + (total_gpu_threads)-1)/(total_gpu_threads);
	int totalIter = num_records_per_thread;
	//ShowLog("GPUMapPartitioner:%f totalIter:%d",t2-t1, totalIter);

	for (int iter = 0; iter< totalIter; iter++){

		double t3 = PandaTimer();
		RunGPUMapTasksHost(*pgc, totalIter -1 - iter, totalIter, grids,blocks);
		cudaThreadSynchronize();
		double t4 = PandaTimer();
		size_t total_mem,avail_mem;
		cudaMemGetInfo( &avail_mem, &total_mem );
		//ShowLog("GPU_ID:[%d] RunGPUMapTasks take %f sec at iter [%d/%d] remain %d mb GPU mem processed",
		//	pgc->gpu_id, t4-t3,iter,totalIter, avail_mem/1024/1024);

	}//for
	//ShowLog("GPU_ID:[%d] Done %d Tasks",pgc->gpu_id,pgc->num_input_record);
	return 0;
}//int 


 void PandaMapReduceJob::InitPandaCPUMapReduce()
 {

	this->pCPUContext									= CreatePandaCPUContext();
	this->pCPUContext->input_key_vals.num_input_record	= cpuMapTasks.size();
	this->pCPUContext->input_key_vals.input_keyval_arr	= (keyval_t *)malloc(cpuMapTasks.size()*sizeof(keyval_t));
	this->pCPUContext->num_cpus_cores					= getCPUCoresNum();

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


void PandaMapReduceJob::InitPandaGPUCardMapReduce()
{
	this->pGPUCardContext = CreatePandaGPUCardContext();
	int numGPUCardMapTasks = gpuCardMapTasks.size();
	if (numGPUCardMapTasks <=0){
		ShowLog("there is no GPU Card information\n");
		return;
	}//if

	this->pGPUCardContext->input_key_vals.num_input_record = numGPUCardMapTasks;
	this->pGPUCardContext->input_key_vals.input_keyval_arr = (keyval_t *)malloc(numGPUCardMapTasks * sizeof(keyval_t));

	for (unsigned int i = 0; i < numGPUCardMapTasks; i++) {
		void *key = this->gpuCardMapTasks[i]->key;
		int keySize = this->gpuCardMapTasks[i]->keySize;
		void *val = this->gpuCardMapTasks[i]->val;
		int valSize = this->gpuCardMapTasks[i]->valSize;

		this->pGPUCardContext->input_key_vals.input_keyval_arr[i].key = key;
		this->pGPUCardContext->input_key_vals.input_keyval_arr[i].keySize = keySize;
		this->pGPUCardContext->input_key_vals.input_keyval_arr[i].val = val;
		this->pGPUCardContext->input_key_vals.input_keyval_arr[i].valSize = valSize;
	}//for
}

void PandaMapReduceJob::InitPandaGPUMapReduce()
{

	this->pGPUContext = CreatePandaGPUContext();
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

		this->pNodeContext->sorted_key_vals.sorted_keyvals_arr_len = 0;
		int max_len = 100;														//configurable
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
			ShowError("messager == NULL");
		}
		MPI_Barrier(MPI_COMM_WORLD);

	}//void

  
  
  void PandaMapReduceJob::StartPandaMessageThread()
  {
    MessageThread = new oscpp::Thread(messager);
    MessageThread->start();
  }

  
  //check whether the intermediate task has been send out
  void PandaMapReduceJob::StartPandaPartitionCheckSends(const bool sync)
  {
    std::vector<oscpp::AsyncIORequest * > newReqs;
    for (unsigned int j = 0; j < sendReqs.size(); ++j)
    {
      if (sync) sendReqs[j]->sync();
      if (sendReqs[j]->query()) delete sendReqs[j];
      else                      newReqs.push_back(sendReqs[j]);
    }
    sendReqs = newReqs;
  }

  //The Hash Partition or Shuffle stage at the program.
  //TODO   3/6/2013
  void PandaMapReduceJob::partitionSub(void * const memPool,
                                           void * const keySpace,
                                           void * const valueSpace,
                                           const int numKeys,
                                           const int singleKeySize,
                                           const int singleValSize)
  {
  
  }//void

  void PandaMapReduceJob::partitionChunk(const unsigned int chunkIndex)
  {

  }//void
  
  void PandaMapReduceJob::StartPandaExitMessager()
  {
    if (messager!=NULL) messager->MsgFinish();

    MessageThread->join();
    delete MessageThread;
	ShowLog("MessageThread Join() completed.");

  }//void



  void PandaMapReduceJob::map()
  {	
    StartPandaMessageThread();
    MPI_Barrier(MPI_COMM_WORLD);
    //enqueueAllChunks();
    StartPandaPartitionCheckSends(true);
    //collectVariablesFromMessageAndKill();
    //freeMapVariables();
  }	

  void PandaMapReduceJob::sort()
  {
  }

  void PandaMapReduceJob::reduce()
  {
    //kernelStream = memcpyStream = cudacpp::Stream::nullStream;
    //enqueueReductions();
    //cudacpp::Runtime::sync();
  }

  PandaMapReduceJob::PandaMapReduceJob(int argc,
                                               char ** argv,
                                               const bool accumulateMapResults,
                                               const bool accumulateReduceResults,
                                               const bool syncOnPartitionSends)
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
	  if(this->getEnableCPU())
		addCPUMapTasks(chunk);
	  if(this->getEnableGPU())
		addGPUMapTasks(chunk);
	  if(this->getEnableGPUCard())
		addGPUCardMapTasks(chunk);
  }//void

  void PandaMapReduceJob::addGPUCardMapTasks(panda::Chunk *chunk)
  {
	  void *key		= chunk->getKey();
	  void *val		= chunk->getVal();
	  int keySize	= chunk->getKeySize();
	  int valSize	= chunk->getValSize();
	  MapTask *pMapTask = new MapTask(keySize,key,valSize,val);
	  gpuCardMapTasks.push_back(pMapTask);
  }//void
		
  void PandaMapReduceJob::addCPUMapTasks(panda::Chunk *chunk)
  {
	  void *key		= chunk->getKey();
	  void *val		= chunk->getVal();
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

  void PandaMapReduceJob::StartPandaCPUCombiner()
  {
	  ExecutePandaCPUCombiner(this->pCPUContext);
  }//void

  void PandaMapReduceJob::StartPandaGPUCardCombiner()
  {
	  ExecutePandaGPUCardCombiner(this->pGPUCardContext);
  }//void

  void PandaMapReduceJob::StartPandaGPUCombiner()
  {
	  ExecutePandaGPUCombiner(this->pGPUContext);
  }//void 
			
  void PandaMapReduceJob::StartPandaSortBucket()
  {
	  ExecutePandaSortBucket(this->pNodeContext);
  }//for

  void PandaMapReduceJob::StartPandaCopyRecvedBucketToGPU(int start_task_id, int end_task_id)
  {
	  ShowLog("copy %d chunks to GPU", (end_task_id - start_task_id) );
	  AddReduceTask4GPU(this->pGPUContext,this->pNodeContext, start_task_id, end_task_id);
  }//void

  void PandaMapReduceJob::StartPandaCopyRecvedBucketToGPUCard(int start_task_id, int end_task_id)
  {
	  ShowLog("copy %d chunks to GPUCard", (end_task_id - start_task_id) );
	  AddReduceTaskOnGPUCard(this->pGPUCardContext,this->pNodeContext, start_task_id, end_task_id);
  }//void

  void PandaMapReduceJob::StartPandaCopyRecvedBucketToCPU(int start_task_id, int end_task_id)
  {
	  AddReduceTask4CPU(this->pCPUContext,this->pNodeContext,start_task_id,end_task_id);
  }//void

  int PandaMapReduceJob::GetHash(const char* Key, int KeySize, int commSize)
  {  
        /////FROM : http://courses.cs.vt.edu/~cs2604/spring02/Projects/4/elfhash.cpp
        unsigned long h = 0;
        while(KeySize-- > 0)
        {
                h = (h << 4) + *Key++;
                unsigned long g = h & 0xF0000000L;
                if (g) h ^= g >> 24;
                h &= ~g;
        }//while            
        return (int) ((int)h % commSize);
  }//int

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
			//delete [] keyBuff
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
			this->pNodeContext->buckets.valBuffSize[bucketId]	= valBuffSize;
			//TODO remove valBuff in std::vector
			//delete [] valBuff;
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

	  }//if
  }//void

  void PandaMapReduceJob::PandaPartitionCheckSends(const bool sync)
  {
    std::vector<oscpp::AsyncIORequest * > newReqs;
    for (unsigned int j = 0; j < sendReqs.size(); ++j)
    {
      if (sync) sendReqs[j]->sync();
      if (sendReqs[j]->query()) delete sendReqs[j];
      else    newReqs.push_back(sendReqs[j]);
    }//for
    sendReqs = newReqs;
  }//void

  void PandaMapReduceJob::StartPandaDoPartitionOnCPU(){

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
		int bucketId = GetHash(key,keySize,this->commSize);
		val_t *vals  = sorted_intermediate_keyvals_arr1[i].vals;
		int len = sorted_intermediate_keyvals_arr1[i].val_arr_len;
		for (int j=0;j<len;j++){
			ShowLog("keySize:%d  valSize:%d buckeId:%d",keySize, vals[j].valSize, bucketId);
			PandaAddKeyValue2Bucket(bucketId, (char *)key, keySize,(char *)(vals[j].val),vals[j].valSize);
		}//for
	  }//for
	  //ShowLog("PandaAddKeyValue2Bucket Done\n");
  }//void

  void PandaMapReduceJob::StartPandaCPUReduceTasks(){
	  //ExecutePandaCPUReduceTasks(this->pCPUContext);
  }//void

  void PandaMapReduceJob::StartPandaPartitionSubSendData()
  {

	  

    for (int index = 0; index < commSize; ++index)
    {
	  int curlen	= this->pNodeContext->buckets.counts[index][0];
	  int maxlen	= this->pNodeContext->buckets.counts[index][1];
	  int keySize	= this->pNodeContext->buckets.counts[index][2];
	  int valSize	= this->pNodeContext->buckets.counts[index][3];

	  ShowLog("index:%d curlen:%d maxlen:%d keySize:%d valSize:%d curlen:%d",index,curlen,maxlen,keySize,valSize,curlen);

	  char *keyBuff = this->pNodeContext->buckets.savedKeysBuff[index];
	  char *valBuff = this->pNodeContext->buckets.savedValsBuff[index];

	  int *keySizeArray = this->pNodeContext->buckets.keySize[index];
	  int *valSizeArray = this->pNodeContext->buckets.valSize[index];
	  int *keyPosArray  = this->pNodeContext->buckets.keyPos[index];
	  int *valPosArray  = this->pNodeContext->buckets.valPos[index];
	  
	  int *keyPosKeySizeValPosValSize = (int *)malloc(sizeof(int)*curlen*4);
	  
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

      int i = (index + commRank + commSize - 1) % commSize;
      
      if (keySize+valSize > 0) // it can happen that we send nothing.
      {
        oscpp::AsyncIORequest * ioReq = messager->sendTo(i,
                                                       	keyBuff,
                                                       	valBuff,
						   								keyPosKeySizeValPosValSize,
                                                       	keySize,
                                                     	valSize,
						   								curlen);
        sendReqs.push_back(ioReq);
      }//if
    }
  }//void

  void PandaMapReduceJob::StartPandaGlobalPartition()
  {

	  PandaPartitionCheckSends(false);
	  StartPandaDoPartitionOnCPU();
	  StartPandaPartitionSubSendData();
	ShowLog("StartPandaPartitionSubSendData is done");

      if (syncPartSends) PandaPartitionCheckSends(true);
	ShowLog("PandaPartitionCheckSends is done");
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
	ShowLog("show log 1.0");
	//////////////////////////
	InitPandaRuntime();
	//////////////////////////

	ShowLog("gpuMapTasks:%lu  gpuCardMapTasks:%lu  cpuMapTasks:%lu",gpuMapTasks.size(),gpuCardMapTasks.size(), cpuMapTasks.size());

	if(this->getEnableGPU())
		InitPandaGPUMapReduce();
	if(this->getEnableGPUCard())
		InitPandaGPUCardMapReduce();
	if(this->getEnableCPU())
		InitPandaCPUMapReduce();

	if(this->getEnableGPU())
		StartPandaGPUMapTasks();
	if(this->getEnableGPUCard())
		StartPandaMapTasksOnGPUCard();
	if(this->getEnableCPU())
		StartPandaCPUMapTasks();
	  
	if(this->getEnableGPU())
		StartPandaGPUCombiner();
	if(this->getEnableCPU())
		StartPandaCPUCombiner();
	if(this->getEnableGPUCard())
		StartPandaGPUCardCombiner();

    if (messager != NULL) messager->MsgFinalize();
	
	if(this->getEnableGPU()){
		StartPandaSortGPUResults();			
		StartPandaLocalMergeGPUOutput();	
	}//if
	
	if(this->getEnableCPU()){
		StartPandaSortCPUResults();
	}//if

	if(this->getEnableGPUCard()){
		StartPandaSortGPUCardResults(0,0);
	}

	///////////////////////////
	//	Shuffle Stage Start  //
	///////////////////////////
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	StartPandaGlobalPartition();
	StartPandaPartitionCheckSends(true);
	ShowLog("after StartPandaPartitionCheckSends");

	StartPandaExitMessager();

	ShowLog("after StartPandaExitMessager");	
	/////////////////////////////////
	//	Shuffle Stage Done
	/////////////////////////////////
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	//Copy recved bucket data into sorted array
	StartPandaSortBucket();
	ShowLog("after StartPandaSortBucket");	
	//TODO schedule
	int start_task_id = 0;
	int end_task_id = this->pNodeContext->sorted_key_vals.sorted_keyvals_arr_len;

	if(this->getEnableGPU())
		StartPandaCopyRecvedBucketToGPU(0, 0);
	if(this->getEnableCPU())
		StartPandaCopyRecvedBucketToCPU(0, 0);
	if(this->getEnableGPUCard())
		StartPandaCopyRecvedBucketToGPUCard(0, end_task_id);
	
	//StartPandaAssignReduceTaskToGPU(start_task_id, end_task_id);
	//StartPandaAssignReduceTaskToGPUCard(start_task_id, end_task_id);
	if(this->getEnableGPU())
		StartPandaReduceTasksOnGPU();
	if(this->getEnableGPUCard())
		StartPandaReduceTasksOnGPUCard();

	//StartPandaCPUReduceTasks();

    MPI_Barrier(MPI_COMM_WORLD);

  }
}
