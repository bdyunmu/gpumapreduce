#ifndef __OSCPP_CLOSURE_H__
#define __OSCPP_CLOSURE_H__

namespace oscpp
{

  class Closure
  {
    public:
      Closure();
      virtual ~Closure();

      virtual void execute() = 0;
  };

}

#endif
