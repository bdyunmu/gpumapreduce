#include <oscpp/AsyncFileReader.h>

  class AsyncIORequest;
  class Closure;

namespace oscpp
{
	AsyncFileReader::AsyncFileReader(){
		handle = 0;
		opened = 0;
		offset= 0;

	}//UNSGINED LONG LONG OFFSET
  
	AsyncFileReader::~AsyncFileReader(){
	}//~AsyncFileReader
	  
	
	AsyncIORequest * AsyncFileReader::open(const std::string fileName){
		return 0;
	}//

	AsyncIORequest *AsyncFileReader::close(){
		return 0;
	}

	AsyncIORequest * AsyncFileReader::read(void * const buffer, const size_t size){
		return 0;
	}

	//inline bool AsyncFileReader::isOpen() const { 
	//	return opened; 
	//}
  	
}