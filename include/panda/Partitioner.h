#ifndef __PANDA_PARTITIONER_H__
#define __PANDA_PARTITIONER_H__

#include <panda/PandaCPUConfig.h>
#include <panda/PandaGPUConfig.h>
#include <cudacpp/Stream.h>

namespace panda
{
  class Partitioner
  {
    public:
      Partitioner();
      virtual ~Partitioner();
      //virtual bool canExecuteOnGPU() const = 0;
      //virtual bool canExecuteOnCPU() const = 0;
      virtual int  GetHash(const char* Key, int KeySize, int commRank );
      //virtual void init() = 0;
  };
}

#endif
