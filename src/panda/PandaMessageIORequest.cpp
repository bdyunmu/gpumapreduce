#include <panda/PandaMessageIORequest.h>
#include <oscpp/Thread.h>
#include <Panda.h>

namespace panda
{

  PandaMessageIORequest::PandaMessageIORequest(volatile bool * const pFlag, volatile bool * const pWaiting, const int pByteCount)
    : oscpp::AsyncIORequest(oscpp::AsyncIORequest::REQUEST_TYPE_WRITE)
  {
    flag = pFlag;
    waiting = pWaiting;
    byteCount = pByteCount;
  }//PandaMessageIORequest

  PandaMessageIORequest::PandaMessageIORequest(const PandaMessageIORequest & rhs) : oscpp::AsyncIORequest(oscpp::AsyncIORequest::REQUEST_TYPE_WRITE)
  {
	  this->flag = rhs.flag;
	  this->byteCount = rhs.byteCount;
	  this->waiting = rhs.waiting;
  }//panda

  PandaMessageIORequest & PandaMessageIORequest::operator = (const PandaMessageIORequest & rhs)
  {
	  this->flag = rhs.flag;
	  this->waiting = rhs.waiting;
	  this->byteCount = rhs.byteCount;
	  return *this;
  }//panda

  PandaMessageIORequest::~PandaMessageIORequest()
  {
    sync();
    delete flag;
  }//panda

  bool PandaMessageIORequest::query()
  {
	ShowLog("query:%d",(int)*flag);
    return *flag;
  }//bool

  void PandaMessageIORequest::sync()
  {
	ShowLog("sync cond.lock");
    cond.lockMutex();
    if (*flag)
    {
      cond.unlockMutex();
      return;
    }
    *waiting = true;
	ShowLog("sync cond.wait");
    cond.wait();
    *waiting = false;
    cond.unlockMutex();
  }//void

  bool PandaMessageIORequest::hasError()
  {
    return false;
  }//bool

  int  PandaMessageIORequest::bytesTransferedCount()
  {
    return byteCount;
  }//int

}
