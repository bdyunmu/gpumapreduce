#ifndef __PANDA_Message_H__
#define __PANDA_Message_H__

#include <oscpp/AsyncIORequest.h>
#include <oscpp/Runnable.h>
#include "Panda.h"

namespace panda
{
  class Message : public oscpp::Runnable
  {
    protected:
      	int commSize, commRank;
    public:
      	Message();
      	virtual ~Message();
  	virtual oscpp::AsyncIORequest * sendTo(	const int rank,
                                             	void * const keys,
                                             	void * const vals,
					 	int * const keyPosKeySizeValPosValSize,
                                             	const int keySize,
                                             	const int valSize,
					 	const int maxlen) = 0;

  	virtual void setPnc(panda_node_context *pnc) = 0;
      	virtual void MsgInit();
      	virtual void MsgFinalize() = 0;
      	virtual void run() = 0;
      	virtual oscpp::AsyncIORequest * MsgFinish() = 0;
  };
}

#endif
