#include <panda/Chunk.h>
#include <panda/FileChunk.h>
#include <cudacpp/Stream.h>
#include <oscpp/AsyncFileReader.h>
#include <oscpp/AsyncIORequest.h>

#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace panda
{
  int FileChunk::key = 0;
  void FileChunk::loadData(const bool async)
  {
    data = reinterpret_cast<char * >(malloc(elemSize * numElems));
    reader = new oscpp::AsyncFileReader;
    oscpp::AsyncIORequest * openReq;
    openReq = reader->open(fileName.c_str());
    openReq->sync();
    if (openReq->hasError())
    {
      fprintf(stderr, "Error opening file '%s'.\n", fileName.c_str());
      fflush(stderr);
    }
    delete openReq;

    int soFar = 0;
    while (soFar < numElems)
    {
      int elemsToRead = std::min(numElems - soFar, maxReadSize / elemSize);
      readReqs.push_back(reader->read(data + soFar * elemSize, elemsToRead * elemSize));
      soFar += elemsToRead;
    }

    if (!async) waitForLoad();
  }
  void FileChunk::waitForLoad()
  {
    if (reader == NULL) return;

    oscpp::AsyncIORequest * closeReq;

    for (unsigned int i = 0; i < readReqs.size(); ++i)
    {
      readReqs[i]->sync();
      if (readReqs[i]->hasError())
      {
        fprintf(stderr, "Error reading from file '%s'.\n", fileName.c_str());
        fflush(stderr);
      }
      delete readReqs[i];
    }
    closeReq = reader->close();
    closeReq->sync();
    if (closeReq->hasError())
    {
      fprintf(stderr, "Error closing file '%s'.\n", fileName.c_str());
      fflush(stderr);
    }
    readReqs.clear();

    delete closeReq;
    delete reader;

    loaded = true;
    reader = NULL;
  }

  FileChunk::FileChunk(const std::string & pFileName,
                                 const int pElemSize,
                                 const int pNumElems,
                                 const int pMaxReadSize,
                                 const int readAheadPos)
  {
    loaded = false;
    reader = NULL;
    data = NULL;
    fileName = pFileName;
    elemSize = pElemSize;
    numElems = pNumElems;
    maxReadSize = pMaxReadSize;
    whenToReadAhead = readAheadPos;
    finishLoading();
  }
  FileChunk::~FileChunk()
  {
    waitForLoad();
    if (data != NULL) free(data);
  }//FileChunk

  void FileChunk::finishLoading()
  {
    if (loaded) return;
    if (reader == NULL) loadData(false);
    waitForLoad();
  }
  bool FileChunk::updateQueuePosition(const int newPosition)
  {
    if (newPosition <= whenToReadAhead && reader == NULL)
    {
      loadData(true);
    }
    return true;
  }
  int  FileChunk::getMemoryRequirementsOnGPU() const
  {
    return numElems * elemSize;
  }
  void FileChunk::stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream)
  {
    finishLoading();
    //cudacpp::Runtime::memcpyHtoDAsync(gpuStorage, data, numElems * elemSize, memcpyStream);
	//cudaMemcpyAsync( gpuStorage, data, numElems * elemSize,cudaMemcpyHostToDevice , memcpyStream->getHandle() );

  }
  void FileChunk::finalizeAsync()
  {
  }
}
