#ifndef __PANDA_MAPPER_H__
#define __PANDA_MAPPER_H__

#include <panda/EmitConfiguration.h>
#include <panda/PandaCPUConfig.h>
#include <panda/PandaGPUConfig.h>
#include <cudacpp/Stream.h>

namespace panda
{
  class Chunk;

  class Mapper
  {
    public:
      Mapper();
      virtual ~Mapper();

      virtual EmitConfiguration getEmitConfiguration(panda::Chunk * const chunk) const = 0;
      virtual bool canExecuteOnGPU() const = 0;
      virtual bool canExecuteOnCPU() const = 0;
      virtual void init() = 0;
      virtual void finalize() = 0;
      virtual void executeOnGPUAsync(panda::Chunk * const chunk, PandaGPUConfig & pandaGPUConfig, void * const gpuMemoryForChunk,
                                     cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream) = 0;
      //virtual void executeOnCPUAsync(panda::Chunk * const chunk, GPMRCPUConfig & gpmrCPUConfig) = 0;
  };
}

#endif
