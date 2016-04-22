#include <cudacpp/Stream.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace cudacpp
{
		/// Constructor for stream 0 ***ONLY***.
	  Stream::Stream(const int ignored){}
		/// Disable the copy constructor.
      //inline Stream::Stream(const Stream & rhs) { }
	    /// Disable the operator =
     //inline Stream::Stream & operator = (const Stream & rhs) { return * this; }

	  /// The default constructor. Simply calls cudaStreamCreate.
	  Stream::Stream()
	  {
		  //cudaStreamCreate(&handle);
	  }
      /// Simply calls cudaStreamDestroy on the native handle.
	  Stream::~Stream()
	  {
		  //cudaStreamDestroy(handle);
	  }

	  /// Simply calls cudaStreamSynchronize on the native handle.
	  void Stream::sync()
	  {
		  //cudaStreamSynchronize(handle);
	  }
	  
      /*
       * Simply calls cudaStreamQuery on the native handle.
       *
       * @return True iff cudaStreamQuery(handle) == cudaSuccess, or in other
       * words, if all commands in the stream have executed successfully.
       */
	  bool Stream::query()
	  {
	return false;
		//return (cudaStreamQuery(handle) == cudaSuccess);
	  }//bool

	  Stream* Stream::nullStream = new Stream();
	  
	  /*
       * @return The native stream handle. Public as a side-effect of me hating
       *         the friend keyword.
       */
	  int* Stream::getHandle(){
		  return 0;//handle;
	  }
}
