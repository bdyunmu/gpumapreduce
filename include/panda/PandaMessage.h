#ifndef __PANDA_MESSAGE_H__
#define __PANDA_MESSAGE_H__

#include <mpi.h>
#include <cuda.h>

#include <panda/Message.h>
#include <oscpp/Condition.h>
#include <oscpp/Mutex.h>
#include <panda/PandaMessagePackage.h>
#include <list>
#include <vector>
#include "Panda.h"

namespace panda
{

  class PandaMessage : public Message
  {
    protected:

      	oscpp::Mutex addDataLock;
      	std::vector<PandaMessagePackage *> needsToBeSent;
      	std::vector<PandaMessagePackage *> pendingIO;

      	bool copySendData;
      	bool pollUnsent();
      	void pollPending();

    public:
     	PandaMessage(const bool pCopySendData); 
	virtual ~PandaMessage();

	virtual oscpp::AsyncIORequest    *sendTo(const int rank,
                                             	void * const keys,
                                             	void * const vals,
			 			int * const keyPosKeySizeValPosValSize,
                                             	const int keySize,
                                             	const int valSize,
					 	const int maxlen);

 	virtual void setPnc(panda_node_context *pnc);
      	virtual void MsgInit();
      	virtual void MsgFinalize();
      	virtual void run();
      	virtual oscpp::AsyncIORequest * MsgFinish();
  
  };
}

#endif
