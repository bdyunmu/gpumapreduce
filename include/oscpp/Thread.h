#ifndef __OSCPP_THREAD_H__
#define __OSCPP_THREAD_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cuda.h>
//#include <cuda_runtime.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <stdarg.h>
#include <pthread.h>

namespace oscpp
{
  class Runnable;

  class Thread
  {
    protected:
      void * handle;
      volatile bool running;
      Runnable * runner;

      static void * startThread(void * vself);
    public:
      Thread(Runnable * const pRunner);
      ~Thread();

      static void yield();
      void start();
      void run();
      void join();
      bool isRunning();
  };
}

#endif
