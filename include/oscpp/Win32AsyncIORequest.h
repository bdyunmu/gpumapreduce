#ifndef __OSCPP_WIN32ASYNCIOREQUEST_H__
#define __OSCPP_WIN32ASYNCIOREQUEST_H__

#ifdef _WIN32

#include <oscpp/AsyncIORequest.h>

#define WIN32_LEAN_AND_MEAN

#include <windows.h>

namespace oscpp
{
  class Win32AsyncIORequest : public AsyncIORequest
  {
    protected:
      int numBytes;
      bool done, error;
      OVERLAPPED overlapped;
      HANDLE fileHandle, event;
    public:
      Win32AsyncIORequest(const int pReqType, HANDLE pFileHandle, const bool alreadyFinished = false);
      virtual ~Win32AsyncIORequest();

      virtual bool query();
      virtual void sync();
      virtual bool hasError();
      virtual int bytesTransferedCount();

      inline OVERLAPPED & getOverlappedStructure()  { return overlapped;          }
      inline HANDLE     & getEvent()                { return event;               }
      inline void         raiseError()              { error = true;               }
  };
}

#endif // _WIN32

#endif
