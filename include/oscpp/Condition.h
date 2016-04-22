#ifndef __OSCPP_CONDITION_H__
#define __OSCPP_CONDITION_H__

#include <oscpp/Mutex.h>

namespace oscpp
{
  class Condition
  {
    protected:
      oscpp::Mutex mutex;
      void * handle;
    public:
      Condition();
      ~Condition();

      void signal();
      void broadcast();
      void wait();
      void lockMutex();
      void unlockMutex();
  };
}

#endif
