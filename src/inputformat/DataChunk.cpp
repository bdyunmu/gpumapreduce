#include <panda/DataChunk.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace panda
{

	
  DataChunk::DataChunk(int mykey, void * const pData,
                        	const int pDataSize)
  {
    key = mykey;
    data = pData;
    dataSize = pDataSize;
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
    return 0;
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
