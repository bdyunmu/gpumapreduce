#ifndef __PANDA_SORTER_H__
#define __PANDA_SORTER_H__

#include <panda/PandaCPUConfig.h>
#include <panda/PandaGPUConfig.h>
#include <cuda.h>

namespace panda
{
  class Sorter
  {
    public:
      Sorter();
      virtual ~Sorter();
      virtual int cpu_compare(const void *key0,int key0_size, const void *key1,int key1_size);
      virtual bool canExecuteOnGPU();
      virtual bool canExecuteOnCPU();
      virtual void init();
      virtual void finalize();
      virtual void executeOnGPUAsync(void *const keys,void *const vals,const int numKeys,int &numUniqueKeys,int **keyOffsets,int **valOffsets,int **numVals);
      virtual void executeOnCPUAsync(void *const keys,void *const vals,const int numKeys,int &numUniqueKeys,int **keyOffsets,int **valOffsets,int **numVals);
      virtual __device__ int gpu_compare(const void *key_a, int len_a, const void *key_b, int len_b);
  };
}

#endif
