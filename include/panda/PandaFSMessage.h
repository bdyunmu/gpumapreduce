#ifndef __PANDA_FSMessage_H__
#define __PANDA_FSMessage_H__

#include <mpi.h>
#include <panda/Message.h>
#include <panda/PandaMessage.h>
#include <oscpp/Condition.h>
#include <oscpp/Mutex.h>

#include <list>
#include <vector>
#include "Panda.h"

namespace panda
{

  class PandaFSMessage : public PandaMessage
  {										
										
    protected:
      oscpp::Mutex addDataLock;
 	oscpp::Condition cond;

	  panda_node_context *pnc;
		
      std::list<PandaMessagePackage * > needsToBeSent;
      std::list<PandaMessagePackage * > pendingIO;
		
      bool innerLoopDone;
      bool copySendData;
	  bool getSendData;
		
	  char inputDataPath[128];
	  char inputStatusPath[128];
		
	  bool pollUnsent();
      void pollPending();
      void pollSends();
		
	  void poll(int & finishedWorkers,
                bool * const workerDone,
                bool * const recvingCount,
                int * const counts,
                int ** keyRecv,
                int ** valRecv,
				int ** keyPosKeySizeValPosValSize,
                MPI_Request * recvReqs);

	  virtual oscpp::AsyncIORequest * sendTo(const int rank,
                                             void * const keys,
                                             void * const vals,
											 int * const keyPosKeySizeValPosValSize,
                                             const int keySize,
                                             const int valSize,
											 const int maxlen);

	  public:
		PandaFSMessage(const bool pCopySendData = false);
		virtual ~PandaFSMessage();
		virtual void run();
		virtual void MsgInit();
		
		virtual void MsgFinalize();
		virtual oscpp::AsyncIORequest * MsgFinish();
		virtual void setPnc(panda_node_context *pnc);
		virtual void PandaAddRecvedBucket(char *keyRecv, char *valRecv, int *keyPosKeySizeValPosValSize, int keyBuffSize, int valBuffSize, int maxLen);

  };
}

#endif
