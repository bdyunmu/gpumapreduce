#ifndef __KEY_VALUE_CHUNK_H__
#define __KEY_VALUE_CHUNK_H__

#include <panda/Chunk.h>
//#include <cudacpp/Runtime.h>
#include <stdlib.h>

namespace panda
{

  class KeyValueChunk : public Chunk
  {
	protected:
          const void * data;
	  int dataSize;

	  const void * key;
	  int keySize;

         int elemSize, numElems;
         void * userData;

	public:
		KeyValueChunk(const void * pKey, int keySize, const void *  pData, int dataSize);
		virtual ~KeyValueChunk();

		virtual bool updateQueuePosition(const int newPosition);
		virtual int getMemoryRequirementsOnGPU() const;
		virtual void stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream);
		virtual void finalizeAsync();
		virtual void finishLoading();
		inline void     setUserData(void * const pUserData) { userData = pUserData; }
	
		inline void	*getKey()		{return (void *)key;}
		inline int      getKeySize()	{return keySize;	}
		inline void * getVal()	{return (void *)data;}
		inline int getValSize()	{return dataSize;}

		inline void *   getUserData()     { return userData;  }

  };


}

#endif
