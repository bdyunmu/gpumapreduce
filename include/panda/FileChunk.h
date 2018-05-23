#ifndef __DATA_CHUNK_H__
#define __DATA_CHUNK_H__

#include <panda/Chunk.h>
#include <cudacpp/Stream.h>
#include <oscpp/AsyncFileReader.h>
#include <oscpp/AsyncIORequest.h>
#include <string>
#include <vector>
#include <stdlib.h>

namespace panda
{
  class FileChunk : public Chunk
  {
    protected:
      enum
      {
        QUEUE_READ_AHEAD_POSITION = 10,
        MAX_READ_SIZE = 32 * 1024 * 1024,
      };

      std::string fileName;
      char * data;
      int elemSize, numElems, maxReadSize;
      volatile bool loaded;
      int whenToReadAhead;
      oscpp::AsyncFileReader * reader;
      std::vector<oscpp::AsyncIORequest * > readReqs;
      void loadData(const bool async);
      void waitForLoad();
      static int key;
    public:
      FileChunk(const std::string & pFileName,
                     const int pElemSize,
                     const int pNumElems,
                     const int pMaxReadSize = MAX_READ_SIZE,
                     const int readAheadPos = QUEUE_READ_AHEAD_POSITION);
      virtual ~FileChunk();

      virtual void finishLoading();
      virtual bool updateQueuePosition(const int newPosition);
      virtual int getMemoryRequirementsOnGPU() const;
      virtual void stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream);
      virtual void finalizeAsync();
      inline void * getKey(){
		int *pInt = (int *)malloc(sizeof(int));
		*pInt = key++;
		return (void *)pInt;
	}
      inline int getKeySize(){ return sizeof(int);}
      inline void * getVal(){ return data;}
      inline int getValSize(){return numElems*elemSize;}
 
      inline int      getElementCount() { return numElems;  }
      inline void *   getData()         { return data;      }
  }; }


#endif
