/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.4
	File: Global.h 
	Time: 2012-12-09 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __KMEANSMAPPER_H__
#define __KMEANSMAPPER_H__

#include "Mapper.h"

class KMeansMapper : public panda::Mapper
{
	
  protected:
//    static const int NUM_BLOCKS = 60;
    float * centers;
    int numCenters;
    int numDims;

    float * accumulatedCenters;
    int   * accumulatedTotals;

    float * gpuCenters;
  public:
    KMeansMapper(const int pNumCenters, const int pNumDims, const float * const pCenters);
    virtual ~KMeansMapper();

    //virtual panda::EmitConfiguration getEmitConfiguration(panda::Chunk * const chunk) const;
    virtual bool canExecuteOnGPU() const;
    virtual bool canExecuteOnCPU() const;
    virtual void init();
    virtual void finalize();
	//virtual void executeOnGPUAsync(cgl::panda::Chunk * const chunk, gpu_context & pandaGPUConfig, void * const gpuMemoryForChunk,
    //                               cudacpp::Stream * kernelStream, cudacpp::Stream * memcpyStream);

	virtual void executeOnCPUAsync(panda::Chunk * const chunk, gpu_context & gpmrCPUConfig);

    void setCenters(const float * const pCenters);

    inline float * getGPUCenters() { return gpuCenters; }
    inline int getNumCenters() const { return numCenters; }
    inline int getNumDims() const { return numDims; }
};

#endif
