#include <panda/DataChunk.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace panda
{

	
  int DataChunk::key = 0;

  DataChunk::DataChunk(void * const pData,
                        	const int pElemSize,
                                             const int pNumElems)
  {
    data = pData;
    elemSize = pElemSize;
    numElems = pNumElems;
  }

  DataChunk::~DataChunk()
  {
  }

  
  bool DataChunk::updateQueuePosition(const int newPosition)
  {
    return false;
  }

  int DataChunk::getMemoryRequirementsOnGPU() const
  {
    return elemSize * numElems;
  }

  void DataChunk::stageAsync(void * const gpuStorage, cudacpp::Stream * const memcpyStream)
  {
    	//cudacpp::Runtime::memcpyHtoDAsync(gpuStorage, data, numElems * elemSize, memcpyStream);
	//cudaMemcpyAsync( gpuStorage, data, numElems * elemSize,cudaMemcpyHostToDevice , memcpyStream->getHandle() );
  }

  void DataChunk::finalizeAsync()
  {
  }

  void DataChunk::finishLoading()
  {
  }

}//DataChunk
