#ifndef __PANDA_CHUNK_H__
#define __PANDA_CHUNK_H__

#include <cudacpp/Stream.h>

namespace panda
{
  class Chunk
  {
    public:
      Chunk();
      virtual ~Chunk();

      virtual void finishLoading() = 0;
      virtual bool updateQueuePosition(const int newPosition) = 0;
      virtual int getMemoryRequirementsOnGPU() const = 0;
      virtual void stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream) = 0;
      virtual void finalizeAsync() = 0;

      //virtual void getSplit() = 0;
      virtual void* getKey() = 0;
      virtual int getKeySize() = 0;
      virtual void* getVal() = 0;
      virtual int getValSize() = 0;

  };

  class MapTask
  {
  public:
	  int keySize;
	  int valSize;
	  void *key;
	  void *val;

	  MapTask();
	  MapTask(int,void*,int, void*);
	  virtual ~MapTask();

  };

}

#endif
