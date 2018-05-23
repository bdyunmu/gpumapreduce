#include <panda/KeyValueChunk.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace panda
{
  KeyValueChunk::KeyValueChunk(const void * pKey, int keySize, const void *  pData, int dataSize)
  {
		this->data = pData;
		this->dataSize = dataSize;
		this->keySize = keySize;
		this->key = pKey;
  }//KeyValueChunk

  KeyValueChunk::~KeyValueChunk()
  {
  }

  int KeyValueChunk::getMemoryRequirementsOnGPU() const
  {
	return dataSize;
  }

  void KeyValueChunk::stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream)
  {
  	//cudacpp::Runtime::memcpyHtoDAsync(gpuStorage, data, dataSize, memcpyStream);
	//cudaMemcpyAsync(gpuStorage,data,dataSize,cudaMemcpyHostToDevice,memcpyStream->getHandle());
  }
	
  void KeyValueChunk::finalizeAsync()
  {
  }

  void KeyValueChunk::finishLoading()
  {
  }

  bool KeyValueChunk::updateQueuePosition(const int newPosition)
  {return false;}
}
