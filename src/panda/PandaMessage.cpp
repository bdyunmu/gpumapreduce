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
	}

	void PandaMessage::pollPending()
	{
	}//void

	PandaMessage::PandaMessage(const bool pCopySendData){
		copySendData = pCopySendData;
	}

	PandaMessage::~PandaMessage()
	{
	}//PandaMessage

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

	void PandaMessage::MsgInit()
	{
		Message::MsgInit();
	}//void

	void PandaMessage::MsgFinalize()
	{
	}//void

	void PandaMessage::run()
	{
	}//void

	oscpp::AsyncIORequest * PandaMessage::MsgFinish()
	{
	}//oscpp
}
