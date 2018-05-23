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
      int elemSize, numElems;
      void * userData;
      static int key;
    public:
      DataChunk(void * const pData,
                              const int pElemSize,
                              const int pNumElems);
      virtual ~DataChunk();

      virtual bool updateQueuePosition(const int newPosition);
      virtual int getMemoryRequirementsOnGPU() const;
      virtual void stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream);
      virtual void finalizeAsync();
      virtual void finishLoading();

      inline void* getKey() 
      {
		  int *pInt = (int *)(malloc(sizeof(int))); 
		  *pInt = key++;
		  return pInt;
      };
      inline int	getKeySize()	{return sizeof(int);};
      inline void*	getVal()		{return data;};
      inline int	getValSize()	{return elemSize*numElems;};
				   
      inline void     setUserData(void * const pUserData) { userData = pUserData; }
      inline int      getElementCount() { return numElems;  }
      inline void *   getData()         { return data;      }
      inline void *   getUserData()     { return userData;  }

  };


}

#endif
