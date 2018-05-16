#ifndef __PANDA_REDUCER_H__
#define __PANDA_REDUCER_H__

#include <panda/EmitConfiguration.h>
#include <panda/PandaGPUConfig.h>
#include <cudacpp/Stream.h>

namespace panda
{
  class Reducer
  {
    public:
      Reducer();
      virtual ~Reducer();
      virtual panda::EmitConfiguration getEmitConfiguration(const void * const keys,
                                                           const int * const numVals,
                                                           const int numKeys,
                                                           int & numKeysToProcess) = 0;
      virtual bool canExecuteOnGPU() const = 0;
      virtual bool canExecuteOnCPU() const = 0;
      virtual void init() = 0;
      virtual void finalize() = 0;
      virtual void executeOnGPUAsync(const int numKeys,
                                     const void * const keys,
                                     const void * const vals,
                                     const int * const keyOffsets,
                                     const int * const valOffsets,
                                     const int * const numVals,
                                     PandaGPUConfig & gpuConfig,
                                     cudacpp::Stream * const kernelStream) = 0;
  };
}
#endif
