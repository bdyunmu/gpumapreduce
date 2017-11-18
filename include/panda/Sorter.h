#ifndef __GPMR_SORTER_H__
#define __GPMR_SORTER_H__

#include <panda/PandaCPUConfig.h>
#include <panda/PandaGPUConfig.h>

namespace panda
{
  class Sorter
  {
    public:
      Sorter();
      virtual ~Sorter();
	  

      virtual bool canExecuteOnGPU() const = 0;
      virtual bool canExecuteOnCPU() const = 0;
      virtual void init() = 0;
      virtual void finalize() = 0;
      virtual void executeOnGPUAsync(void * const keys, void * const vals, const int numKeys, int & numUniqueKeys, int ** keyOffsets, int ** valOffsets, int ** numVals) = 0;
      virtual void executeOnCPUAsync(void * const keys, void * const vals, const int numKeys, int & numUniqueKeys, int ** keyOffsets, int ** valOffsets, int ** numVals) = 0;
  };
}

#endif
