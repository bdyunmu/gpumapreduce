#ifndef __PANDA_MAPREDUCEJOB_H__
#define __PANDA_MAPREDUCEJOB_H__

#include <panda/Message.h>
#include <panda/EmitConfiguration.h>
#include <panda/PandaGPUConfig.h>
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
	void AddReduceTask4GPU(panda_gpu_context* pgc, panda_node_context *pnc, int start_row_id, int end_row_id);	
	void AddReduceTask4CPU(panda_cpu_context* pgc, panda_node_context *pnc, int start_row_id, int end_row_id);
	void StartPandaCPUReduceTasks();
	void StartPandaCPUCombiner();
	void StartPandaSortCPUResults();
 	void StartPandaCopyRecvedBucketToCPU(int, int);
	bool getEnableGPU() {return enableGPU;}
	bool getEnableCPU() {return enableCPU;}
        void setEnableCPU(bool b) {enableCPU = b;}
	void setEnableGPU(bool b) {enableGPU = b;}

	bool enableGPU;
	bool enableCPU;
      	void * keys;
      	void * vals;
      	int keySize, keySpace;
      	int valSize, valSpace;
      	int numUniqueKeys;
      	int * keyOffsets;
      	int * valOffsets;
      	int * numVals;

      	oscpp::Thread * MessageThread;
      	cudacpp::Stream * kernelStream, * memcpyStream;

      	//variables for both map and reduce
      	void * cpuKeys, * cpuVals, * gpuKeys, * gpuVals;
      	int maxStaticMem, maxKeySpace, maxValSpace, numBuffers;
      	std::vector<EmitConfiguration> emitConfigs;
	  
	  panda_gpu_context *pGPUContext;
	  panda_cpu_context *pCPUContext;
	  panda_node_context *pNodeContext;
	  panda_runtime_context *pRuntimeContext;

      //map variables
      void * gpuStaticMems;
      int * cpuKeyOffsets, * cpuValOffsets, * gpuKeyOffsets, * gpuValOffsets;
      int * cpuKeyCounts, * cpuValCounts, * gpuKeyCounts, * gpuValCounts;
      bool accumMap;
      bool syncPartSends;
      std::vector<void * > savedKeys, savedVals;
      std::vector<int> keyAndValCount;
      //std::vector<oscpp::AsyncIORequest * > sendReqs;

      //reduce variables
      std::vector<PandaGPUConfig> configs;
      std::vector<int> keyCount;
      bool accumReduce;
      int maxInputKeySpace, maxInputValSpace, maxInputValOffsetSpace, maxInputNumValsSpace;
      void * gpuInputKeys, * gpuInputVals;
      int  * gpuInputValOffsets, * gpuInputValCounts;

      // timing variables
      oscpp::Timer fullMapTimer, fullReduceTimer, fullTimer;
      oscpp::Timer mapPostTimer, mapFreeTimer;
      oscpp::Timer mapTimer;
      oscpp::Timer binningTimer;
      oscpp::Timer sortTimer;
      oscpp::Timer reduceTimer;
      oscpp::Timer totalTimer;
      
      	virtual int StartPandaReduceTasksOnGPU();
      	virtual void StartPandaMessageThread();
	  
	  virtual void InitPandaRuntime();
	  virtual void InitPandaGPUMapReduce();
	  virtual void StartPandaGPUCombiner();
	  virtual void StartPandaSortGPUResults();
	  virtual void StartPandaLocalMergeGPUOutput();
	  virtual void StartPandaGlobalPartition();
	  virtual void StartPandaDoPartitionOnCPU();
	  virtual void StartPandaSortBucket();

	  virtual int  StartPandaGPUMapTasks();
	  virtual int  StartPandaCPUMapTasks();
	  virtual void PandaPartitionCheckSends(const bool sync);
	  virtual void StartPandaPartitionSendData();
	  virtual void StartPandaCopyRecvedBucketToGPU(int,int);
	  virtual void WaitPandaMessagerExit();
	  virtual void InitPandaCPUMapReduce();

	  virtual int  GetHash(const char* Key, int KeySize, int commRank );
	  virtual void PandaAddKeyValue2Bucket(int bucketId, const char*key, int keySize, const char*val, int valSize);

    public:
      PandaMapReduceJob(int argc,
                            char ** argv,
                            const bool accumulateMapResults     = false,
                            const bool accumulateReduceResults  = false,
                            const bool syncOnPartitionSends     = true);
      ~PandaMapReduceJob();

        virtual void addInput(panda::Chunk * chunk);
	virtual void addCPUMapTasks(panda::Chunk *chunk);
	virtual void addGPUMapTasks(panda::Chunk *chunk);
        virtual void execute();
  };
}

#endif
