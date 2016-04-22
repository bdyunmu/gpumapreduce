#ifndef __CUDACPP_STREAM_H__
#define __CUDACPP_STREAM_H__

#include <driver_types.h>


namespace cudacpp
{
  /**
   * The Stream class encapsulates a cudaStream_t object. The stream object is
   * automatically created within the constructor, so it is important to set the
   * CUDA device before creating a stream. Unexpected behavior may result if a
   * stream is created for one device and used on another. //CUDA Context
   *
   * For more details, please consult the CUDA reference manual.
   */
  class Stream
  {
    protected:
      //The pointer to the cudaStream_t.
      //cudaStream_t handle;

      /// Constructor for stream 0 ***ONLY***.
      Stream(const int ignored);

      /// Disable the copy constructor.
      inline Stream(const Stream & rhs) { }

      /// Disable the operator =
      inline Stream & operator = (const Stream & rhs) { return * this; }
    public:
      /// The default constructor. Simply calls cudaStreamCreate.
      Stream();
      /// Simply calls cudaStreamDestroy on the native handle.
      ~Stream();

      /// Simply calls cudaStreamSynchronize on the native handle.
      void sync();

      /**
       * Simply calls cudaStreamQuery on the native handle.
       *
       * @return True iff cudaStreamQuery(handle) == cudaSuccess, or in other
       * words, if all commands in the stream have executed successfully.
       */
      bool query();

      /**
       * @return The native stream handle. Public as a side-effect of me hating
       *         the friend keyword.
       */
      int * getHandle();

      /// A freely usable handle to stream 0
      static Stream* nullStream;
  };
}

#endif
