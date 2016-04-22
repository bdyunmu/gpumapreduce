
#ifndef __PANDA_FIXEDSIZEMessageIOREQUEST_H__
#define __PANDA_FIXEDSIZEMessageIOREQUEST_H__

#include <oscpp/AsyncIORequest.h>
#include <oscpp/Condition.h>

namespace panda
{

  class PandaMessageIORequest : public oscpp::AsyncIORequest
  {

	protected:

      volatile bool * flag;
      volatile bool * waiting;

      int byteCount;
      oscpp::Condition cond;

      PandaMessageIORequest(const PandaMessageIORequest & rhs);
      PandaMessageIORequest & operator = (const PandaMessageIORequest & rhs);

    public:

      PandaMessageIORequest(volatile bool * const pFlag, volatile bool * const pWaiting, const int pByteCount);
      virtual ~PandaMessageIORequest();

      virtual bool query();
      virtual void sync();
      virtual bool hasError();
      virtual int bytesTransferedCount();
      inline oscpp::Condition & condition() { return cond; }
		
  };
}


#endif
