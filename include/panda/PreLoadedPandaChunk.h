#ifndef __PRELOADEDFIXEDSIZECHUNK_H__
#define __PRELOADEDFIXEDSIZECHUNK_H__

#include <panda/Chunk.h>
//#include <cudacpp/Runtime.h>
#include <stdlib.h>

namespace panda
{
  class PreLoadedPandaChunk : public Chunk
  {
    protected:
      void * data;
      int elemSize, numElems;
      void * userData;
	  static int key;
    public:
      PreLoadedPandaChunk(void * const pData,
                              const int pElemSize,
                              const int pNumElems);
      virtual ~PreLoadedPandaChunk();

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


  class VariousSizePandaChunk : public Chunk
  {
	protected:
      const void * data;
	  int dataSize;

	  const void * key;
	  int keySize;

      int elemSize, numElems;
      void * userData;

	public:
		VariousSizePandaChunk(const void * pKey, int keySize, const void *  pData, int dataSize);
		virtual ~VariousSizePandaChunk();

		virtual bool updateQueuePosition(const int newPosition);
		virtual int getMemoryRequirementsOnGPU() const;
		virtual void stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream);
		virtual void finalizeAsync();
		virtual void finishLoading();

		inline void     setUserData(void * const pUserData) { userData = pUserData; }
		inline void		* getKey()		{return (void *)key;}
		inline int      getKeySize()	{return keySize;	}

		inline int		getDataSize()	{return dataSize;}
		inline void		* getData()		{return (void *)data;}

		inline void * getVal()	{return (void *)data;}
		inline int getValSize()	{return dataSize;}

		inline void *   getUserData()     { return userData;  }

  };


}

#endif
