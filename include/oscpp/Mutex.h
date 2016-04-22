#ifndef __OSCPP_MUTEX_H__
#define __OSCPP_MUTEX_H__

namespace oscpp
{
  class Mutex
  {
    protected:
      void * handle;
    public:
      Mutex();
      ~Mutex();

      bool tryLock();
      void lock();
      void unlock();
      void * getHandle();
  };
}

#endif
