#ifndef __OSCPP_GENERICASYNCIOREQEUST_H__
#define __OSCPP_GENERICASYNCIOREQEUST_H__

#ifndef _WIN32

  #include <oscpp/AsyncIORequest.h>
  #include <cstdio> // for size_t

  namespace oscpp
  {
    class GenericAsyncIORequest : public AsyncIORequest
    {
      protected:
        void * buffer;
        size_t size;
        volatile bool done, error;
      public:
        GenericAsyncIORequest(const int pReqType, void * const buf, const size_t size);

        virtual ~GenericAsyncIORequest();

        virtual bool query();
        virtual void sync();
        virtual bool hasError();
        virtual int bytesTransferedCount();

        inline void   setDone()     { done = true;  }
        inline void   raiseError()  { error = true; }
        inline void * getBuffer()   { return buffer; }
    };
  }

#endif // ifndef _WIN32

#endif
