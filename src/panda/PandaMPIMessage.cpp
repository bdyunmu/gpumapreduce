/***********************************************************************
Copyright 2012 The Trustees of Indiana University.  All rights reserved.
Panda:a MapReduce Framework on GPUs and CPUs

File: PandaMPIMessage.cpp
Time: 2017-12-11

Developer: Hui Li (huili@ruijie.com.cn)

************************************************************************/

#include <mpi.h>
#include <panda/PandaMessageIORequest.h>
#include <panda/PandaMPIMessage.h>
#include <cstring>
#include "Panda.h"

namespace panda
{

	PandaMPIMessage::PandaMPIMessage() : PandaMessage(false)
	{
	}//void

	PandaMPIMessage::PandaMPIMessage(const bool pCopySendData) : PandaMessage(pCopySendData)
	{
		//copySendData  = pCopySendData;
	}//void

	PandaMPIMessage::~PandaMPIMessage()
	{
	}//void

	void PandaMPIMessage::setPnc(panda_node_context *pnc)
	{
		this->pnc = pnc;
	}//void

	void PandaMPIMessage::MsgInit(){
		Message::MsgInit();
		this->innerLoopDone = false;
		this->copySendData = false;
		this->getSendData = false;
		this->numSentBuckets = 0;
	}//void

	void PandaMPIMessage::MsgFinalize(){
	
	}//void

	oscpp::AsyncIORequest * PandaMPIMessage::MsgFinish()
	{
		//return sendTo(commRank, NULL, NULL, NULL, -1, -1, -1);
	}//oscpp

	void PandaMPIMessage::run()
	{

		ShowLog("start PandaMPIMessage thread. run() thread [%d/%d]", commRank, commSize);
		
		int finishedWorkers	= 0;
		bool  * workerDone					= new bool[commSize];
		int   * counts						= new int [commSize*3];
		int  ** keyRecv						= new int*[commSize];
		int  ** valRecv						= new int*[commSize];
		int  ** keyPosKeySizeValPosValSize	= new int*[commSize];
		memset(counts,0,sizeof(int)*commSize*3);
	
		if(workerDone==NULL)		ShowLog("Error");
		if(counts==NULL)		ShowLog("Error");
		if(keyRecv==NULL)		ShowLog("Error");
		if(valRecv==NULL)		ShowLog("Error");
		if(keyPosKeySizeValPosValSize == NULL)	ShowLog("Error");

		MPI_Request * dataReqs  = new MPI_Request[commSize*3];
		MPI_Request * countReqs = new MPI_Request[commSize];
		MPI_Status reqStats[3];

		for (int i = 0; i < commSize; ++i)
		{
			workerDone[i]   = false;
			keyRecv[i] 		= NULL;
			valRecv[i] 		= NULL;
			keyPosKeySizeValPosValSize[i] = NULL;
			MPI_Irecv(counts+i*3,3,MPI_INT,i,0,MPI_COMM_WORLD,(MPI_Request *)&(countReqs[i]));
		}//for

		while (numSentBuckets < commSize || finishedWorkers < commSize)
		{

			//ShowLog("I am looping numSentBuckets:%d finishedWorkers:%d commSize:%d",numSentBuckets,finishedWorkers,commSize);
			const int MAX_SENDS_PER_LOOP = 30;
			int index = 0;
			while (index<MAX_SENDS_PER_LOOP && pollUnsent())
			{
				index++;
			}//while
			//pollPending();
			for (int i=0; i<commSize; i++)
			{
				if (workerDone[i])
					continue;

			      	MPI_Wait(&(countReqs[i]),NULL);
				ShowLog("->recved countReqs from %d, maxlen:%d keysize:%d valsize:%d",
							i, counts[i*3+0], counts[i*3+1], counts[i*3+2]);
				//received null chunk
				if(counts[i*3+0] == -1 && counts[i*3+1] == -1 && counts[i*3+2] == -1){
					workerDone[i] = true;
					finishedWorkers++;
					continue;
				}//if
					
				keyPosKeySizeValPosValSize[i] = new int[4*counts[i*3+0]];
				keyRecv[i] = new int[(counts[i*3+1] + sizeof(int)-1)/sizeof(int)];
				valRecv[i] = new int[(counts[i*3+2] + sizeof(int)-1)/sizeof(int)];

				MPI_Irecv((char*)(keyRecv[i]),counts[i*3+1],MPI_CHAR,i,1,MPI_COMM_WORLD,dataReqs+i*3+1);
				MPI_Irecv((char*)(valRecv[i]),counts[i*3+2],MPI_CHAR,i,2,MPI_COMM_WORLD,dataReqs+i*3+2);
				MPI_Irecv(keyPosKeySizeValPosValSize[i],4*counts[i*3+0],MPI_INT,i,3,MPI_COMM_WORLD,dataReqs+i*3+0);

				MPI_Wait(dataReqs+i*3+1, &reqStats[1]);
				MPI_Wait(dataReqs+i*3+2, &reqStats[2]);
				MPI_Wait(dataReqs+i*3+0, &reqStats[0]);

				if ((counts[0]>0)&&(counts[1]>0)&&(counts[2]>0))
				PandaAddRecvedBucket((char *)keyRecv[i],(char *)valRecv[i],keyPosKeySizeValPosValSize[i], 
					counts[i*3+1],counts[i*3+2], counts[i*3+0]);

				workerDone[i] = true;
				finishedWorkers++;
			}
			pollPending();
		}//while
		ShowLog("Message Looping Done (numSentBuckets:%d finishedWorkers:%d commSize:%d)",numSentBuckets,finishedWorkers,commSize);
		//MPI_Waitall(commSize, zeroReqs, MPI_STATUSES_IGNORE);
		/*
		delete [] workerDone;
		delete [] recvingCount;
		delete [] counts;
		delete [] keyRecv;
		delete [] valRecv;
		delete [] dataReqs;
		*/
	}


	oscpp::AsyncIORequest * PandaMPIMessage::sendTo(const int rank,
		void * const keys,
		void * const vals,
		int * const keyPosKeySizeValPosValSize,
		const int keySize,
		const int valSize,
		const int maxlen) //curlen
	{
		PandaMessagePackage * data  = new PandaMessagePackage;
		data->counts = new int[3];
		data->flag	= new volatile bool;
		data->waiting	= new volatile bool;
		*data->flag	= false;
		*data->waiting	= false;
		data->reqs 	= new MPI_Request[4];
		data->stats 	= new MPI_Status[4];
		if (copySendData)
		{
			if (keySize > 0 && keys != NULL)
			{
				data->keysBuff = new char[keySize];
				memcpy(data->keysBuff, keys, keySize);
			}//if
			else
			{
				data->keysBuff = keys;
			}//else

			if (valSize > 0 && vals != NULL)
			{
				data->valsBuff = new char[valSize];
				memcpy(data->valsBuff, vals, valSize);
			}//if
			else
			{
				data->valsBuff = vals;
			}//else

			if (maxlen > 0 && keyPosKeySizeValPosValSize != NULL)
			{
				data->keyPosKeySizeValPosValSize = new int[4*maxlen];
				memcpy(data->keyPosKeySizeValPosValSize, keyPosKeySizeValPosValSize, 4*maxlen*sizeof(int));
			}
			else
			{
				data->keyPosKeySizeValPosValSize = keyPosKeySizeValPosValSize;
			}//if
		}
		else
		{
			data->keysBuff				= keys;
			data->valsBuff				= vals;
			data->keyPosKeySizeValPosValSize	= keyPosKeySizeValPosValSize;
		}//else
		data->keyBuffSize	= keySize;
		data->valBuffSize	= valSize;
		data->rank		= rank;

		/*if (rank == commRank)
		{
			data->counts[0] = maxlen;
			data->counts[1] = keySize;
			data->counts[2] = valSize;
			data->done[0]   = false; 
			data->done[1]   = false;
			data->done[2]   = false;
			data->done[3]   = false;
		}*/
 
			data->counts[0] = maxlen;
			data->counts[1] = keySize;
			data->counts[2] = valSize;
			data->done[0]   = false;
			data->done[1]   = false;
			data->done[2]   = false;
			data->done[3]   = false;

		PandaMessageIORequest * req = new PandaMessageIORequest(data->flag, data->waiting,
			4*data->counts[0]*sizeof(int) + data->counts[1] + data->counts[2]);
		data->cond = &req->condition();
		addDataLock.lock();
		needsToBeSent.push_back(data);
		addDataLock.unlock();
		return req;
	}


	bool PandaMPIMessage::pollUnsent()
		{

		PandaMessagePackage * data = NULL;
		do {
			addDataLock.lock();
			if (!needsToBeSent.empty())
			{
			data = needsToBeSent.front();
			needsToBeSent.erase(needsToBeSent.begin());
			}//if
			addDataLock.unlock();

			if (data != NULL){
				this->getSendData = true;
				break;
			}//if
		} while((data == NULL)&&(!this->getSendData));

		if (data == NULL)
			return false;

		ShowLog("start to send out a data from %d to %d . data: curlen:%d  keySize:%d  valSize:%d",
					commRank, data->rank, data->counts[0], data->counts[1], data->counts[2]);

		/*if (data->rank == commRank)
		{
			data->cond->lockMutex();
			if (*data->waiting) {
				data->cond->broadcast();
			} //if
			*data->flag = true;
			data->cond->unlockMutex();
		}*///if
		
		if(data->counts[0]==-1 && data->counts[1]==-1 && data->counts[2]==-1){
		MPI_Isend(data->counts,3,MPI_INT,data->rank,0,MPI_COMM_WORLD,&data->reqs[0]);
		}
		if(data->counts[0]>0 && data->counts[1]>0 && data->counts[2]>0){
		MPI_Isend(data->counts,3,MPI_INT,data->rank,0,MPI_COMM_WORLD,&data->reqs[0]);
		MPI_Isend(data->keysBuff,data->keyBuffSize, MPI_CHAR, data->rank, 1, MPI_COMM_WORLD, &data->reqs[1]);
               	MPI_Isend(data->valsBuff,data->valBuffSize, MPI_CHAR, data->rank, 2, MPI_COMM_WORLD, &data->reqs[2]);
	       	MPI_Isend(data->keyPosKeySizeValPosValSize, data->counts[0]*4, MPI_INT, data->rank, 3, MPI_COMM_WORLD, &data->reqs[3]);
		}
		pendingIO.push_back(data);
		return true;
	}

	void PandaMPIMessage::pollPending()
	{
		if (pendingIO.empty()) 
			return;
		std::vector<PandaMessagePackage * > newPending;
		for (std::vector<PandaMessagePackage * >::iterator it=pendingIO.begin(); it!=pendingIO.end();)
		{
			PandaMessagePackage *data = *it;
			int flag = 0;
			if(data->counts[0]>0&&data->counts[1]>0&&data->counts[2]>0){
				MPI_Testall(4, data->reqs, &flag, data->stats);
			}
			if(data->counts[0]==-1&&data->counts[1]==-1&&data->counts[2]==-1)
				MPI_Testall(1, data->reqs, &flag, data->stats);
			if (flag)
			{
				data->cond->lockMutex();
				if (*data->waiting) data->cond->broadcast();
				*data->flag = true;
				data->cond->unlockMutex();
				pendingIO.erase(it);
				if (copySendData)
				{
					//if (data->keysBuff != NULL) delete [] reinterpret_cast<char * >(data->keysBuff);
					//if (data->valsBuff != NULL) delete [] reinterpret_cast<char * >(data->valsBuff);
					//if (data->keyPosKeySizeValPosValSize != NULL) delete [] reinterpret_cast<char *>(data->keyPosKeySizeValPosValSize);
				}//if
				delete data;
				this->numSentBuckets++;
			}//if
			else 
			{
				newPending.push_back(data);
				it++;
			}//else
		}//for
		pendingIO = newPending;
	}//void

	void PandaMPIMessage::PandaAddRecvedBucket(char *keyRecv, char *valRecv, int *keyPosKeySizeValPosValSize, int keyBuffSize, int valBuffSize, int maxlen)
	{

		//keyRecv[i], valRecv[i], keyPosKeySizeValPosValSize[i], counts[i*3+1], counts[i*3+2], counts[i*3+2])
		if (maxlen>0)
		{
		char *newKeyRecv = new char[keyBuffSize];
		memcpy(newKeyRecv, keyRecv, keyBuffSize);
		pnc->recv_buckets.savedKeysBuff.push_back(newKeyRecv);

		char *newValRecv = new char[valBuffSize];
		memcpy(newValRecv, valRecv, valBuffSize);
		pnc->recv_buckets.savedValsBuff.push_back(newValRecv);

		int *keyPosArray = new int[maxlen];
		memcpy(keyPosArray, keyPosKeySizeValPosValSize, maxlen*sizeof(int));
		pnc->recv_buckets.keyPos.push_back(keyPosArray);

		int *keySizeArray = new int[maxlen];
		memcpy(keySizeArray, keyPosKeySizeValPosValSize+maxlen, maxlen*sizeof(int));
		pnc->recv_buckets.keySize.push_back(keySizeArray);

		int *valPosArray = new int[maxlen];
		memcpy(valPosArray, keyPosKeySizeValPosValSize+2*maxlen, maxlen*sizeof(int));
		pnc->recv_buckets.valPos.push_back(valPosArray);

		int *valSizeArray = new int[maxlen];
		memcpy(valSizeArray, keyPosKeySizeValPosValSize+3*maxlen, maxlen*sizeof(int));
		pnc->recv_buckets.valSize.push_back(valSizeArray);

		int *counts = new int[3];
		counts[0] = maxlen;
		counts[1] = keyBuffSize;
		counts[2] = valBuffSize;
		pnc->recv_buckets.counts.push_back(counts);
		}//int

	}

}
