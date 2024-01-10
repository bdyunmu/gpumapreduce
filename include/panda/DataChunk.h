#ifndef __DATA_CHUNK_H__
#define __DATA_CHUNK_H__

#include <panda/Chunk.h>
#include <stdlib.h>

namespace panda
{
  class DataChunk : public Chunk
  {
    protected:
      void * data;
      int dataSize;
      int key;
    public:
      DataChunk(int mykey, void * const pData,
                              const int pDataSize);

      virtual ~DataChunk();

      virtual bool updateQueuePosition(const int newPosition);
      virtual int getMemoryRequirementsOnGPU() const;
      virtual void stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream);
      virtual void finalizeAsync();
      virtual void finishLoading();

      inline int 	getKey() 	{return key;};
      inline int	getKeySize()	{return sizeof(int);};
      inline void*	getData()	{return data;};
      inline int	getDataSize()	{return dataSize;};

  };


}
#endif
