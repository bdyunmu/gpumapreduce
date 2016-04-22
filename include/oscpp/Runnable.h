#ifndef __OSCPP_RUNNABLE_H__
#define __OSCPP_RUNNABLE_H__

namespace oscpp
{
  class Runnable
  {
    public:
      Runnable();
      virtual ~Runnable();

      virtual void run() = 0;
  };
}

#endif
