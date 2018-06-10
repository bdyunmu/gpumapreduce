#ifndef __PANDA_MAPREDUCEJOB_H__
#define __PANDA_MAPREDUCEJOB_H__

#include <panda/Message.h>
#include <panda/EmitConfiguration.h>
#include <panda/MapReduceJob.h>
#include <cudacpp/Stream.h>
#include <oscpp/Thread.h>
#include <oscpp/Timer.h>
#include <vector>
#include "Panda.h"

namespace panda
{

  class Chunk;
  class MapTask;
  class ReduceTask;
  class EmitConfiguration;
	
  class PandaMapReduceJob : public MapReduceJob
  {
	
    protected:
	
        std::vector<Chunk * >   chunks;
	std::vector<MapTask *>  mapTasks;
	std::vector<MapTask *>	cpuMapTasks;
	std::vector<MapTask *>	gpuMapTasks;
	std::vector<ReduceTask *> reduceTasks;

	void addCPUMapTasks(panda::Chunk *chunk);
	void addGPUMapTasks(panda::Chunk *chunk);
	void AddGPUReduceTask(panda_gpu_context* pgc, panda_node_context *pnc, int start_id, int end_id);	
	void AddCPUReduceTask(panda_cpu_context* pgc, panda_node_context *pnc, int start_id, int end_id);

	void InitPandaCPUContext();
	void InitPandaGPUContext();

	void StartPandaMapTasksSchedule();	
	void StartPandaCPUMapTasks();
	void StartPandaCPUReduceTasks();
	void StartPandaCPUCombineTasks();
	void StartPandaCPUSortTasks();
	void StartPandaCPUDumpReduceTasks();

 	void StartPandaCopyRecvedBucketToCPU(int, int);
	
	bool getEnableGPU() {return enableGPU;}
	bool getEnableCPU() {return enableCPU;}
        //void setEnableCPU(bool b) {enableCPU = b;}
	//void setEnableGPU(bool b) {enableGPU = b;}
	int  taskLevel;
	//bool setTaskLevel(int tl) {taskLevel = tl;return true;}

	bool enableGPU;
	bool enableCPU;

      	oscpp::Thread * MessageThread;
      	cudacpp::Stream * kernelStream, * memcpyStream;
      	std::vector<EmitConfiguration> emitConfigs;
	  
	panda_gpu_context *pGPUContext;
	panda_cpu_context *pCPUContext;
	panda_node_context *pNodeContext;
	panda_runtime_context *pRuntimeContext;

        // timing variables
        oscpp::Timer mapTimer;
        oscpp::Timer reduceTimer;
      	
	void StartPandaGPUMapTasks();
      	void StartPandaGPUReduceTasks();
     	void StartPandaGPUSortTasks();
	void StartPandaGPUCombineTasks();

 	virtual void StartPandaMessageThread();
	  
	virtual void InitPandaNodeContextAndRuntime();
	virtual void InitPandaGPUMapReduce();
	virtual void InitPandaCPUMapReduce();

	virtual void StartPandaLocalMergeGPUOutput();
	virtual void StartPandaGlobalPartition();
	virtual void StartPandaDoPartitionOnCPU();
	virtual void StartPandaSortBucket();

	virtual void StartPandaPartitionSendData();
	virtual void StartPandaCopyRecvedBucketToGPU(int,int);
	virtual void WaitPandaMessagerExit();

	virtual void PandaAddKeyValue2Bucket(int bucketId, const char*key, int keySize, const char*val, int valSize);

    public:
        PandaMapReduceJob(int argc, char **argv);
        ~PandaMapReduceJob();
	bool setTaskLevel(int tl) {taskLevel = tl;return true;}
	void setEnableCPU(bool b) {enableCPU = b;}
	void setEnableGPU(bool b) {enableGPU = b;}
        virtual void addInput(panda::Chunk * chunk);
        virtual void execute();

  };
}

#endif
