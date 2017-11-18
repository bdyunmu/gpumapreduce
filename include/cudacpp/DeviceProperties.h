#ifndef __CUDACPP_DEVICEPROPERTIES_H__
#define __CUDACPP_DEVICEPROPERTIES_H__


#include <cudacpp/myString.h>
#include <cudacpp/Vector3.h>
#include <sys/types.h>


namespace cudacpp
{
  /**
   * This class abstracts away a call to cudaGetDeviceProperties, and keeps
   * a variable-lifetime-persistent copy of the return value.
   */
  class DeviceProperties
  {
    protected:
      /// The name of the device.
      String name;
      /// The maximum number of threads in a single block.
      int maxThreadsPerBlock;
      /// The maximum number of threads in each dimension.
      Vector3<int> maxThreadsDim;
      /// The maximum number of blocks in each dimension.
      Vector3<int> maxGridSize;
      /// The total amount of shared memory available to a SM.
      int sharedMemPerBlock;
      /// The SIMD size of each SM.
      int warpSize;
      /// The pitch of global device memory.
      int memPitch;
      /// The total number of 32-bit registers per SM.
      int regsPerBlock;
      /// The frequency of the clock, in kHz.
      int clockRate;
      /// The device's compute capability major number.
      int major;
      /// The device's compute capability minor number.
      int minor;
      /// The number of SMs on the device.
      int multiProcessorCount;
      /// The total amount of constant memory available to a kernel.
      size_t totalConstantMemory;
      /// The total amount of global device memory.
      size_t totalMemBytes;
      /// The alignment requirements for textures in memory.
      size_t textureAlign;
      /// Default constructor.
     
    public:
      /**
       * Creates a new structure with the specified member values. This is used
       * primarily in conjunction with Runtime::chooseDevice.
       */
      static DeviceProperties * create( const String & pName,
                                        const int maxThreadsPerBlock,
                                        const Vector3<int> & pMaxThreadsDim,
                                        const Vector3<int> & pMaxGridSize,
                                        const int pSharedMemPerBlock,
                                        const int pWarpSize,
                                        const int MemPitch,
                                        const int pRegsPerBlock,
                                        const int pClockRate,
                                        const int pMajor,
                                        const int pMinor,
                                        const int pMultiProcessorCount,
                                        const size_t pTotalConstantMemory,
                                        const size_t pTotalMemBytes,
                                        const size_t pTextureAlign);
	   DeviceProperties();
	  ~DeviceProperties();
      /**
       * Queries the runtime for the properties of the specified device.
       *
       * @param   deviceID The CUDA ID of the device to query.
       * @return  A user-deletable pointer to a new device property structure,
       *          provided that the specified device exists. If a CUDA runtime
       *          error is reported, then the return value is NULL.
       */
      static DeviceProperties * get(const int deviceID);
      /// @return The name of this device.
      const String & getName() const;
      /// @return The maximum number of threads in a single block.
      int           getMaxThreadsPerBlock() const;
      /// @return The maximum number of threads in each dimension.
      const Vector3<int> & getMaxBlockSize() const;
      /// @return The maximum number of blocks in each dimension.
      const Vector3<int> & getMaxGridSize() const;
      /// @return The total amount of shared memory available to a SM.
      int           getSharedMemoryPerBlock() const;
      /// @return The total amount of constant memory available to a kernel.
      size_t        getTotalConstantMemory() const;
      /// @return The SIMD size of each SM.
      int           getWarpSize() const;
      /// @return The pitch of global device memory.
      int           getMemoryPitch() const;
      /// @return The total number of 32-bit registers per SM.
      int           getRegistersPerBlock() const;
      /// @return The frequency of the clock, in kHz.
      int           getClockRate() const;
      /// @return The alignment requirements for textures in memory.
      size_t        getTextureAlignment() const;
      /// @return The total amount of global device memory.
      size_t        getTotalMemory() const;
      /// @return The device's compute capability major number.
      int           getMajor() const;
      /// @return The device's compute capability minor number.
      int           getMinor() const;
      /// @return The number of SMs on the device.
      int           getMultiProcessorCount() const;
  };
}

#endif
