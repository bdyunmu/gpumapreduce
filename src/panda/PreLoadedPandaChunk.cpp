#include <panda/PreLoadedPandaChunk.h>
//#include <cudacpp/Runtime.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace panda
{

	
  int PreLoadedPandaChunk::key = 0;

  PreLoadedPandaChunk::PreLoadedPandaChunk(void * const pData,
                                            const int pElemSize,
                                             const int pNumElems)
  {
    data = pData;
    elemSize = pElemSize;
    numElems = pNumElems;
  }

  PreLoadedPandaChunk::~PreLoadedPandaChunk()
  {
  }

  
  bool PreLoadedPandaChunk::updateQueuePosition(const int newPosition)
  {
    return false;
  }

  int PreLoadedPandaChunk::getMemoryRequirementsOnGPU() const
  {
    return elemSize * numElems;
  }

  void PreLoadedPandaChunk::stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream)
  {
    //cudacpp::Runtime::memcpyHtoDAsync(gpuStorage, data, numElems * elemSize, memcpyStream);
//	cudaMemcpyAsync( gpuStorage, data, numElems * elemSize,cudaMemcpyHostToDevice , memcpyStream->getHandle() );
  }

  void PreLoadedPandaChunk::finalizeAsync()
  {
  }

  void PreLoadedPandaChunk::finishLoading()
  {

  }

}//PreLoadedPandaChunk

namespace panda
{

	VariousSizePandaChunk::VariousSizePandaChunk(const void * pKey, int keySize, const void *  pData, int dataSize)
	{
		this->data = pData;
		this->dataSize = dataSize;
		this->keySize = keySize;
		this->key = pKey;
	}//VariousSizePandaChunk

	VariousSizePandaChunk::~VariousSizePandaChunk()
	{
	}

	int VariousSizePandaChunk::getMemoryRequirementsOnGPU() const
	{
		return dataSize;
	}

	void VariousSizePandaChunk::stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream)
	{
    	//cudacpp::Runtime::memcpyHtoDAsync(gpuStorage, data, dataSize, memcpyStream);
	//cudaMemcpyAsync(gpuStorage,data,dataSize,cudaMemcpyHostToDevice,memcpyStream->getHandle());
	}
	
	void VariousSizePandaChunk::finalizeAsync()
	{
	}

	void VariousSizePandaChunk::finishLoading()
	{

	}

	bool VariousSizePandaChunk::updateQueuePosition(const int newPosition)
	{return false;}
}
