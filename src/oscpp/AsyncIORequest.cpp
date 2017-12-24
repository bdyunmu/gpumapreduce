#include <oscpp/AsyncIORequest.h>

namespace oscpp
{
	AsyncIORequest::AsyncIORequest(const int pReqType){
	} //

	AsyncIORequest::AsyncIORequest(const AsyncIORequest & rhs){
	}

	//AsyncIORequest::AsyncIORequest & operator = (const AsyncIORequest & rhs){
	//	
	//}//AsyncIORequest::AsyncIORequest need to be updated.  

	AsyncIORequest::~AsyncIORequest(){
	}

	bool AsyncIORequest::query(){
		return false;
	}//

	void AsyncIORequest::sync(){

	}

	bool AsyncIORequest::hasError(){
	return false;
	}

	int AsyncIORequest::bytesTransferedCount() {
		return 0;
	}

	/*inline int AsyncIORequest::getRequestType() const { 
		int reqType = 0;
		return reqType; 
	}*/

}//


  
