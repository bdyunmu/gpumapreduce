#ifndef __OSCPP_ASYNCIOREQUEST_H__
#define __OSCPP_ASYNCIOREQUEST_H__

namespace oscpp
{
  class AsyncIORequest
  {
    public:
      enum
      {
        REQUEST_TYPE_OPEN = 0,
        REQUEST_TYPE_CLOSE,
        REQUEST_TYPE_READ,
        REQUEST_TYPE_WRITE,
      };
    protected:
      int reqType;
      AsyncIORequest(const int pReqType);
      AsyncIORequest(const AsyncIORequest & rhs);
      AsyncIORequest & operator = (const AsyncIORequest & rhs);
    public:
      virtual ~AsyncIORequest();
      virtual bool query() = 0;
      virtual void sync() = 0;
      virtual bool hasError() = 0;
      virtual int bytesTransferedCount() = 0;
      inline int getRequestType() const { return reqType; }
  };
}

#endif
