#ifndef __PANDA_FIXEDSIZEMessage_H__
#define __PANDA_FIXEDSIZEMessage_H__

#include <mpi.h>
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

      	bool innerLoopDone;
      	bool copySendData;
      	int zeroCount[2];
      	std::vector<MPI_Request> zeroReqs;

      	int singleKeySize, singleValSize;
      	char * finalKeys, * finalVals;
      	int finalKeySize, finalValSize;
      	int finalKeySpace, finalValSpace;

      	bool pollUnsent();
      	void pollPending();
      	void pollSends();
      	void poll(int & finishedWorkers,
                bool * const workerDone,
                bool * const recvingCount,
                int * const counts,
                int ** keyRecv,
                int ** valRecv,
                MPI_Request * recvReqs);
      	void privateAdd(const void * const keys, const void * const vals, const int keySize, const int valSize);
      	void grow(const int size, const int finalSize, int & finalSpace, char *& finals);

    public:
      	PandaMessage(const int pSingleKeySize, const int pSingleValSize, const bool pCopySendData = false);
     	PandaMessage(const bool pCopySendData); 
	virtual ~PandaMessage();

      	virtual oscpp::AsyncIORequest * sendTo(const int rank,
                                             void * const keys,
                                             void * const vals,
                                             const int keySize,
                                             const int valSize);

	virtual oscpp::AsyncIORequest * sendTo(const int rank,
                                             void * const keys,
                                             void * const vals,
			 int * const keyPosKeySizeValPosValSize,
                                             const int keySize,
                                             const int valSize,
					 const int maxlen);

      	virtual oscpp::AsyncIORequest * sendTo(const int rank,
                                             void * const keys,
                                             void * const vals,
                                             int * const keySizes,
                                             int * const valSizes,
                                             const int numKeys,
                                             const int numVals);

 	virtual void setPnc(panda_node_context *pnc);
      	virtual void MsgInit();
      	virtual void MsgFinalize();
      	virtual void run();
      	virtual oscpp::AsyncIORequest * MsgFinish();
      	virtual void getFinalDataSize(int & keySize, int & valSize) const;
      	virtual void getFinalDataSize(int & keySize, int & valSize, int & numKeys, int & numVals) const;
      	virtual void getFinalData(void * keyStorage, void * valStorage) const;
      	virtual void getFinalData(void * keyStorage, void * valStorage, int * keySizes, int * valSizes) const;
  };
}

#endif
