#ifndef __OSCPP_ASYNCFILEREADER_H__
#define __OSCPP_ASYNCFILEREADER_H__

#include <string>

namespace oscpp
{
  class AsyncIORequest;
  class Closure;
  class AsyncFileReader
  {
    protected:
      void * handle;
      bool   opened;
      unsigned long long offset;
    public:
      AsyncFileReader();
      ~AsyncFileReader();
	  
	  AsyncIORequest * open(const std::string fileName);
      AsyncIORequest * close();
      AsyncIORequest * read(void * const buffer, const size_t size);

      inline bool isOpen() const { return opened; }
  };
}

#endif
