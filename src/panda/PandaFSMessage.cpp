/*

Copyright 2013 The Trustees of Indiana University.  All rights reserved.
MapReduce Framework on GPUs and CPUs
Code Name: Panda 0.43
File: PandaMessage.cpp
Time: 2013-07-11
Developer: Hui Li (lihui@indiana.edu)

This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

*/

#include <mpi.h>
#include <panda/PandaMessageIORequest.h>
#include <panda/PandaFSMessage.h>
#include <cstring>
#include "Panda.h"
#include <stdio.h>

namespace panda
{

	PandaFSMessage::PandaFSMessage(const bool pCopySendData) : PandaMessage(pCopySendData)
	{

	  //copySendData  = pCopySendData;
	

	}//void

	PandaFSMessage::~PandaFSMessage(){

	}//void

	void PandaFSMessage::setPnc(panda_node_context *pnc)
	{
	  this->pnc = pnc;
	  
	}//void

	void PandaFSMessage::MsgInit(){

	Message::MsgInit();
	this->innerLoopDone = false;
	this->copySendData	= false;
	this->getSendData	= false;
	sprintf(inputDataPath,"C:\\OpenMPI_v1.6.2-x64\\Panda2\\data\\N%d\\",commRank);
	sprintf(inputStatusPath,"C:\\OpenMPI_v1.6.2-x64\\Panda2\\data\\N%d\\",commRank);
	ShowLog("inputDataPath:%s		len:%zu",inputDataPath,strlen(inputDataPath));
	ShowLog("inputStatusPath:%s	len:%zu",inputStatusPath,strlen(inputStatusPath));

	}//void

	void PandaFSMessage::MsgFinalize(){
	
	}//void

	oscpp::AsyncIORequest * PandaFSMessage::MsgFinish()
	{
		return sendTo(commRank, NULL, NULL, NULL, -1, -1, -1);
	}//oscpp

	void PandaFSMessage::run()
	{

		ShowLog("start PandaFSMessage thread to run at [%d/%d]", commRank, commSize);
		int		finishedWorkers	= 0;
		bool	* workerDone					= new bool[commSize];
		bool	* recvingCount					= new bool[commSize];
		int		* counts						= new int [commSize * 3];
		int		* zeroCounts					= new int [commSize * 3];
		int		** keyRecv						= new int*[commSize];
		int		** valRecv						= new int*[commSize];
		int		** keyPosKeySizeValPosValSize	= new int*[commSize];
		
		if(workerDone==NULL)					ShowLog("Error");
		if(recvingCount==NULL)					ShowLog("Error");
		if(counts==NULL)						ShowLog("Error");
		if(keyRecv==NULL)						ShowLog("Error");
		if(valRecv==NULL)						ShowLog("Error");
		if(keyPosKeySizeValPosValSize == NULL)	ShowLog("Error");

		//commRank/fromRank/keySizeValSizeMaxlen
		//			/keyBuff
		//			/valBuff
		//			/keyPosKeySizeValPosValSize
		//keySize/ValSize/maxlen
		
		for (int i = 0; i < commSize; ++i)
		{
			workerDone[i]		= false;
			recvingCount[i]		= true;
			keyRecv[i] 			= NULL;
			valRecv[i] 			= NULL;
			keyPosKeySizeValPosValSize[i] = NULL;
			//ShowLog("counts[0]:%d counts[1]:%d counts[2]:%d",counts[0],counts[1],counts[2]);
		}//for
		//MPI_Barrier(MPI_COMM_WORLD);
		
		int flag;
		innerLoopDone 	= false;

		while (!innerLoopDone || finishedWorkers < commSize)
		{
			const int MAX_SENDS_PER_LOOP = 10;
			int index = 0;
			while (++index < MAX_SENDS_PER_LOOP && pollUnsent())
			{
				//sleep(100);
			}//while
			pollPending();

			for (int i = 0; i < commSize; ++i)
			{
				if (workerDone[i])
					continue;

				if (i != commRank){

				//check  //commRank/fromRank/data
				keyPosKeySizeValPosValSize[i] = new int[4*counts[i * 3 + 0]];
				keyRecv[i] = new int[(counts[i * 3 + 1] + sizeof(int) - 1) / sizeof(int)];
				valRecv[i] = new int[(counts[i * 3 + 2] + sizeof(int) - 1) / sizeof(int)];

			//MPI_Irecv(counts + i * 3, 3, MPI_INT, i, 0, MPI_COMM_WORLD, (MPI_Request *)&(countReqs[i]));
			/*	MPI_Irecv((char*)(keyRecv[i]), counts[i * 3 + 1], MPI_CHAR, i, 1, MPI_COMM_WORLD, recvReqs + i * 3 + 1);
				MPI_Irecv((char*)(valRecv[i]), counts[i * 3 + 2], MPI_CHAR, i, 2, MPI_COMM_WORLD, recvReqs + i * 3 + 2);
				MPI_Irecv(keyPosKeySizeValPosValSize[i], 4*counts[i * 3 + 0], MPI_INT, i, 3, MPI_COMM_WORLD, recvReqs + i * 3 + 0);*/

				PandaAddRecvedBucket((char *)keyRecv[i], (char *)valRecv[i], keyPosKeySizeValPosValSize[i], counts[i * 3 + 1], counts[i * 3 + 2], counts[i * 3 + 0]);

				}

				++finishedWorkers;
				workerDone[i] = true;
				//MPI_Irecv(zeroCounts + i * 3, 3, MPI_INT, i, 4, MPI_COMM_WORLD, zeroReqs + i );
			}

		}//while

		//MPI_Waitall(commSize, zeroReqs, MPI_STATUSES_IGNORE);
		ShowLog("MPI_Waitall done finsihedWorkers:%d  commSize:%d", finishedWorkers, commSize);
		/*
		delete [] workerDone;
		delete [] recvingCount;
		delete [] counts;
		delete [] keyRecv;
		delete [] valRecv;
		delete [] recvReqs;
		*/
		//ShowLog("Delete array buffer");

	}

	//replace with ADIOS interface
	//write through MPI send and receive
	oscpp::AsyncIORequest * PandaFSMessage::sendTo(const int rank,
		void * const keys,
		void * const vals,
		int * const keyPosKeySizeValPosValSize,
		const int keySize,
		const int valSize,
		const int maxlen)
	{
		PandaMessagePackage * data  = new PandaMessagePackage;

		data->flag		= new volatile bool;
		data->waiting	= new volatile bool;
		*data->flag		= false;
		*data->waiting	= false;

		//write to disk for fault tolerance
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
			data->keysBuff						= keys;
			data->valsBuff						= vals;
			data->keyPosKeySizeValPosValSize	= keyPosKeySizeValPosValSize;
		}//else

		data->keyBuffSize	= keySize;
		data->valBuffSize	= valSize;
		data->rank			= rank;

		if (rank == commRank)
		{
			data->counts[0] = maxlen;
			data->counts[1] = keySize;
			data->counts[2] = valSize;
			//data->done[0]   = data->done[1] = data->done[2] = data->done[3] = false;
		} //if
		else
		{
			data->counts[0] = maxlen;
			data->counts[1] = keySize;
			data->counts[2] = valSize;
			data->done[0]   = data->done[1] = data->done[2] = data->done[3] = false;
		} //else

		PandaMessageIORequest * req = new PandaMessageIORequest(data->flag, data->waiting, 
			4*data->counts[0]*sizeof(int) + data->counts[1] + data->counts[2]);
		data->cond = &req->condition();
		addDataLock.lock();
		needsToBeSent.push_back(data);
		addDataLock.unlock();
		return req;
	}

	bool PandaFSMessage::pollUnsent()
	{
		int zeroCount[3];
		zeroCount[0] = zeroCount[1] = zeroCount[2] = 0;
		PandaMessagePackage * data = NULL;

		do {
			addDataLock.lock();
			if (!needsToBeSent.empty())
			{
			data = needsToBeSent.front();
			needsToBeSent.pop_front();
			}//if
			addDataLock.unlock();

			if (data != NULL){
				this->getSendData = true;
				break;
			}//if
			//sleep(100);
		} while((data == NULL)&&(!this->getSendData));

		if (data == NULL)
			return false;

		ShowLog("start to send out a data from %d to %d  data: maxlen:%d  keySize:%d  valSize:%d",
			commRank, data->rank, data->counts[0], data->counts[1], data->counts[2]);

		if (data->rank == commRank)
		{
			if (data->counts[0] == 0)
			{
				//ShowLog("data->counts[0] == 0 add null data to local bucket");
				PandaAddRecvedBucket((char *)(data->keysBuff), (char *)(data->valsBuff), data->keyPosKeySizeValPosValSize, data->keyBuffSize, data->valBuffSize,data->counts[0]);
			}
			else if (data->counts[0] > 0)
			{
				//ShowLog("data->counts[0] > 0 add data to local bucket");
				PandaAddRecvedBucket((char *)(data->keysBuff), (char *)(data->valsBuff), data->keyPosKeySizeValPosValSize, data->keyBuffSize, data->valBuffSize,data->counts[0]);
			}	//if
			else if (data->counts[0] < 0)
			{
				//ShowLog("data->counts[0] < 0 MPI_Isend(zeroCount,3,MPI_INT,i,4,MPI_COMM_WORLD");
				innerLoopDone = true;
				//finish the messager data->counts[0] == -1
				//The difference between SendTo(NULL,NULL,NULL,-1,-1,-1) is that there is no need to creat data object
				for (int i = 0; i < commSize; i++)
				{
					//the current messager is exiting and notify the other process prepare for exiting as well.
					//send zero count to ask receiver to exit the processing for peroformance issue.
					FILE * pFile;
					//MPI_Isend(zeroCount, 3, MPI_INT, i, 4, MPI_COMM_WORLD, &zeroReqs[i]);
					
					char str[32];
					sprintf(str, "\\N%d\\status.txt", i);
					strcat(this->inputStatusPath,str);
					ShowLog("%s",this->inputStatusPath);
					pFile = fopen(this->inputStatusPath,"w+");

					if (pFile == NULL) {
						ShowError("Error opening file");
						return false;
					}
					else
					{
						fwrite(zeroCount, sizeof(int), 3, pFile);
						fflush(pFile);
						ShowLog("write zeroCount to :%s",this->inputStatusPath);
						fclose (pFile);
					}//else
					//there is no need to push to pendingIO.
				}//for
			}//else

			data->cond->lockMutex();
			if (*data->waiting) {
				data->cond->broadcast();
				//data is in buff, there is no need to wait
			} //if
			*data->flag = true;
			data->cond->unlockMutex();
			//reserve the data to avoid segment fault?
			
		}//if
		else
		{
			
			//send data asynchronizally, and put it in the pending task queue
			if(data->counts[0] <= 0)
				ShowError("!  data->counts[0] <= 0");
			if(data->keyBuffSize != data->counts[1])
				ShowError("!  data->keyBuffSize != data->counts[1]");
			if(data->valBuffSize != data->counts[2])
				ShowError("!  data->valBuffSize != data->counts[2]");

			if(data->counts[0] > 0){

			FILE * pFile;
			char str[32];
			sprintf(str, "\\N%d\\data.txt", data->rank);
			strcat(this->inputDataPath,str);
			ShowLog("%s",this->inputDataPath);
			pFile = fopen(this->inputDataPath,"w+");

			if (pFile == NULL)
			{
				ShowError ("Error opening file");
				return false;
			}
			else
			{
				fwrite(data->counts, sizeof(int), 3, pFile);
				fwrite(data->keysBuff, sizeof(char), data->keyBuffSize, pFile);
				fwrite(data->valsBuff, sizeof(char), data->valBuffSize, pFile);
				fwrite(data->keyPosKeySizeValPosValSize, sizeof(int), data->counts[0]*4, pFile);
				//todo above	
				fflush(pFile);
				fclose (pFile);
			}
			//MPI_Isend(data->counts,      3,    MPI_INT,     data->rank,  0,  MPI_COMM_WORLD, &data->reqs[0]);
			//MPI_Isend(data->keysBuff,    data->keyBuffSize, MPI_CHAR, data->rank, 1, MPI_COMM_WORLD, &data->reqs[1]);
			//MPI_Isend(data->valsBuff,    data->valBuffSize, MPI_CHAR, data->rank, 2, MPI_COMM_WORLD, &data->reqs[2]);
			//MPI_Isend(data->keyPosKeySizeValPosValSize, data->counts[0]*4, MPI_INT, data->rank, 3, MPI_COMM_WORLD, &data->reqs[3]);
			pendingIO.push_back(data);
			}

		} //else
		return true;
	}

	void PandaFSMessage::pollPending()
	{
		if (pendingIO.empty())
			return;

		std::list<PandaMessagePackage * > newPending;
		for (std::list<PandaMessagePackage * >::iterator it = pendingIO.begin(); it != pendingIO.end(); ++it)
		{
			PandaMessagePackage * data = *it;
			int flag = 0;
			
			FILE * pFile;
			char str[32];
			sprintf(str, "\\N%d\\data.txt", data->rank);
			strcat(this->inputDataPath,str);
			//ShowLog(this->inputDataPath);
			pFile = fopen(this->inputDataPath,"r");

			if(pFile!=NULL){
				flag = 1;
				fclose(pFile);
			}else
				flag = 0;

			if (flag)
			{
				data->cond->lockMutex();
				if (*data->waiting) data->cond->broadcast();
				*data->flag = true;
				data->cond->unlockMutex();
				//pendingIO.remove();
				//the data object has been sent out
				if (copySendData)
				{
					//if (data->keysBuff != NULL) delete [] reinterpret_cast<char * >(data->keysBuff);
					//if (data->valsBuff != NULL) delete [] reinterpret_cast<char * >(data->valsBuff);
					//if (data->keyPosKeySizeValPosValSize != NULL) delete [] reinterpret_cast<char *>(data->keyPosKeySizeValPosValSize);
				}//if
				//delete data;
			}//if
			else
			{
				newPending.push_back(data);
			}//else
		}//for

		pendingIO = newPending;

	}//void

	void PandaFSMessage::pollSends()
	{
		const int MAX_SENDS_PER_LOOP = 10;
		int index = 0;
		while (++index < MAX_SENDS_PER_LOOP && pollUnsent()) 
		{
			//sleep(200);
			ShowLog("loop pollUnsent and sleep 200ms");
		}
		index = 0;
		pollPending();
	}//void

	void PandaFSMessage::PandaAddRecvedBucket(char *keyRecv, char *valRecv, int *keyPosKeySizeValPosValSize, int keyBuffSize, int valBuffSize, int maxlen)
	{	
		//keyRecv[i], valRecv[i], keyPosKeySizeValPosValSize[i], counts[i * 3 + 1], counts[i * 3 + 2], counts[i * 3 + 2])
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
		}

	}

	void PandaFSMessage::poll(int & finishedWorkers,
		bool * const workerDone,
		bool * const recvingCount,
		int *  const counts,
		int ** keyRecv,
		int ** valRecv,
		int ** keyPosKeySizeValPosValSize,
		MPI_Request * recvReqs)
	{
	}
	
}
