/*

Copyright 2013 The Trustees of Indiana University.  All rights reserved.
Panda: a MapReduce Framework on GPUs and CPUs
File: PandaMessage.cpp
Time: 2013-07-11
Developer: Hui Li (huili@ruijie.com.cn)

*/

#include <mpi.h>
#include <panda/PandaMessageIORequest.h>
#include <panda/PandaMessage.h>
#include <cstring>
#include "Panda.h"

namespace panda
{

	void PandaMessage::setPnc(panda_node_context *pnc){
	}//void

	bool PandaMessage::pollUnsent()
	{
		PandaMessagePackage * data = NULL;
		addDataLock.lock();
		if (!needsToBeSent.empty())
		{
			data = needsToBeSent.front();
			needsToBeSent.erase(needsToBeSent.begin());
		}//if
		addDataLock.unlock();
		if (data == NULL) return false;

		if (data->rank == commRank)
		{
			if (data->keySize != -1)
			{
				privateAdd(data->keys, data->vals, data->keySize, data->valSize);
			}//if
			else
			{
				innerLoopDone = true;
				for (int i = 0; i < commSize; ++i)
				{
					MPI_Isend(zeroCount, 2, MPI_INT, i, 0, MPI_COMM_WORLD, &zeroReqs[i]);
				}//for
			}//else
			data->cond->lockMutex();
			if (*data->waiting) data->cond->broadcast();
			*data->flag = true;
			data->cond->unlockMutex();
			delete data;
		}//if
		else
		{
			MPI_Isend(data->counts,             2, MPI_INT,  data->rank, 0, MPI_COMM_WORLD, &data->reqs[0]);
			MPI_Isend(data->keys,   data->keySize, MPI_CHAR, data->rank, 0, MPI_COMM_WORLD, &data->reqs[1]);
			MPI_Isend(data->vals,   data->valSize, MPI_CHAR, data->rank, 0, MPI_COMM_WORLD, &data->reqs[2]);
			pendingIO.push_back(data);
		}//else
		return true;
	}

	void PandaMessage::pollPending()
	{
		if (pendingIO.empty()) return;
		std::vector<PandaMessagePackage *> newPending;
		for (std::vector<PandaMessagePackage *>::iterator it = pendingIO.begin(); it != pendingIO.end(); ++it)
		{
			PandaMessagePackage * data = *it;
			int flag;

			MPI_Testall(3, data->reqs, &flag, data->stats);
			if (flag)
			{
				data->cond->lockMutex();
				if (*data->waiting) data->cond->broadcast();
				*data->flag = true;
				data->cond->unlockMutex();
				delete [] data->counts;
				if (copySendData)
				{
					if (data->keys != NULL) delete [] reinterpret_cast<char * >(data->keys);
					if (data->vals != NULL) delete [] reinterpret_cast<char * >(data->vals);
				}//if
				delete data;
			}//if
			else
			{
				newPending.push_back(data);
			}//else
		}//for
		pendingIO = newPending;
	}//void

	void PandaMessage::pollSends()
	{
		const int MAX_SENDS_PER_LOOP = 20;
		int index = 0;
		while (++index < MAX_SENDS_PER_LOOP && pollUnsent()) 
		{ 
			//sleep(100);
		}
		index = 0;
		pollPending();
	}//void

	void PandaMessage::poll(   int & finishedWorkers,
		bool * const workerDone,
		bool * const recvingCount,
		int * const counts,
		int ** keyRecv,
		int ** valRecv,
		MPI_Request * recvReqs)
	{
		pollSends();
		int flag;
		MPI_Status stat[2];
		for (int i = 0; i < commSize; ++i)
		{
			if (workerDone[i]) continue;
			if (recvingCount[i])
			{
				MPI_Test(recvReqs + i * 2, &flag, stat);
				if (flag)
				{
					// printf("%2d - recv'd counts %d and %d from %d.\n", commRank, counts[i * 2 + 0], counts[i * 2 + 1], i); fflush(stdout);
					recvingCount[i] = false;
					if (counts[i * 2] == 0)
					{
						workerDone[i] = true;
						++finishedWorkers;
					}
					else
					{
						keyRecv[i] = new int[counts[i * 2 + 0] / sizeof(int)];
						valRecv[i] = new int[counts[i * 2 + 1] / sizeof(int)];
						MPI_Irecv(keyRecv[i], counts[i * 2 + 0], MPI_CHAR, i, 0, MPI_COMM_WORLD, recvReqs + i * 2 + 0);
						MPI_Irecv(valRecv[i], counts[i * 2 + 1], MPI_CHAR, i, 0, MPI_COMM_WORLD, recvReqs + i * 2 + 1);
					}//else
				}//if
			}//if
			else
			{
				MPI_Testall(2, recvReqs + i * 2, &flag, stat);
				if (flag)
				{
					privateAdd(keyRecv[i], valRecv[i], counts[i * 2 + 0], counts[i * 2 + 1]);
					recvingCount[i] = true;
					MPI_Irecv(counts + i * 2, 2, MPI_INT, i, 0, MPI_COMM_WORLD, recvReqs + i * 2);
					delete [] keyRecv[i];
					delete [] valRecv[i];
				}//MPI_Testall
			}//else
		}
	}

	void PandaMessage::grow(const int size, const int finalSize, int & finalSpace, char *& finals)
	{
		if (size + finalSize > finalSpace)
		{
			int newSpace = finalSpace * 2;
			while (size + finalSize > newSpace) newSpace *= 2;
			finalSpace = newSpace;
			char * temp = new char[finalSpace];
			memcpy(temp, finals, finalSize);
			delete [] finals;
			finals = temp;
		}//if
	}//void

	void PandaMessage::privateAdd(const void * const keys, const void * const vals, const int keySize, const int valSize)
	{

		grow(keySize, finalKeySize, finalKeySpace, finalKeys);
		grow(valSize, finalValSize, finalValSpace, finalVals);
		memcpy(finalKeys + finalKeySize, keys, keySize);
		memcpy(finalVals + finalValSize, vals, valSize);
		finalKeySize += keySize;
		finalValSize += valSize;

	}//void
	PandaMessage::PandaMessage(const bool pCopySendData){
		singleKeySize = 0;
		singleValSize = 0;
		copySendData = pCopySendData;
	}
	PandaMessage::PandaMessage(const int pSingleKeySize, const int pSingleValSize, const bool pCopySendData)
	{
		singleKeySize = pSingleKeySize;
		singleValSize = pSingleValSize;
		copySendData  = pCopySendData;
	}//PandaMessage

	PandaMessage::~PandaMessage()
	{

	}//PandaMessage

	oscpp::AsyncIORequest * PandaMessage::sendTo(const int rank,
		void * const keys,
		void * const vals,
		const int keySize,
		const int valSize)
	{
		PandaMessagePackage * data  = new PandaMessagePackage();
		data->flag = new volatile bool;
		data->waiting = new volatile bool;
		*data->flag = false;
		*data->waiting = false;
		if (copySendData)
		{
			if (keySize > 0 && keys != NULL)
			{
				data->keys = new char[keySize];
				memcpy(data->keys, keys, keySize);
			}
			else
			{
				data->keys = keys;
			}
			if (valSize > 0 && vals != NULL)
			{
				data->vals = new char[valSize];
				memcpy(data->vals, vals, valSize);
			}
			else
			{
				data->vals = vals;
			}
		}
		else
		{
			data->keys = keys;
			data->vals = vals;
		}
		data->keySize = keySize;
		data->valSize = valSize;
		data->rank = rank;

		if (rank == commRank)
		{
			data->counts = NULL;
		}
		else
		{
			data->counts = new int[2];
			data->counts[0] = keySize;
			data->counts[1] = valSize;
			data->done[0] = data->done[1] = data->done[2] = false;
		}
		PandaMessageIORequest * req = new PandaMessageIORequest(data->flag, data->waiting, data->keySize + data->valSize);
		data->cond = &req->condition();

		addDataLock.lock();
		needsToBeSent.push_back(data);
		addDataLock.unlock();
		return req;
	}

	oscpp::AsyncIORequest * PandaMessage::sendTo(const int rank,
		void * const keys,
		void * const vals,
		int * const keyPosKeySizeValPosValSize,
		const int keySize,
		const int valSize,
		const int maxlen)
	{
		return NULL;
	}

	oscpp::AsyncIORequest * PandaMessage::sendTo(const int rank,
		void * const keys,
		void * const vals,
		int * const keySizes,
		int * const valSizes,
		const int numKeys,
		const int numVals)
	{
		return NULL;
	}//oscpp

	void PandaMessage::MsgInit()
	{
		Message::MsgInit();
		zeroReqs.resize(commSize);
		zeroCount[0]  = zeroCount[1] = 0;
		finalKeySpace = 1048576;
		finalValSpace = 1048576;
		finalKeySize  = 0;
		finalValSize  = 0;
		finalKeys     = new char[finalKeySpace];
		finalVals     = new char[finalValSpace];
	}//void

	void PandaMessage::MsgFinalize()
	{
		delete [] finalKeys;
		delete [] finalVals;
	}//void

	void PandaMessage::run()
	{
		//run
		int finishedWorkers		= 0;
		bool  * workerDone      = new bool[commSize];
		bool  * recvingCount    = new bool[commSize];
		int   * counts          = new int [commSize * 2];
		int  ** keyRecv         = new int*[commSize];
		int  ** valRecv         = new int*[commSize];
		MPI_Request * recvReqs  = new MPI_Request[commSize * 2];

		for (int i = 0; i < commSize; ++i)
		{
			workerDone[i] = false;
			recvingCount[i] = true;
			keyRecv[i] = NULL;
			valRecv[i] = NULL;
			MPI_Irecv(counts + i * 2, 2, MPI_INT, i, 0, MPI_COMM_WORLD, recvReqs + i * 2);
		}//for

		innerLoopDone = false;
		while (!innerLoopDone || finishedWorkers < commSize)
		{
			poll(finishedWorkers, workerDone, recvingCount, counts, keyRecv, valRecv, recvReqs);
			pollSends();
		}//while
		MPI_Waitall(commSize, &zeroReqs[0], MPI_STATUSES_IGNORE);

		delete [] workerDone;
		delete [] recvingCount;
		delete [] counts;
		delete [] keyRecv;
		delete [] valRecv;
		delete [] recvReqs;

	}//void

	oscpp::AsyncIORequest * PandaMessage::MsgFinish()
	{
		return sendTo(commRank, NULL, NULL, -1, -1);
	}//oscpp


}
