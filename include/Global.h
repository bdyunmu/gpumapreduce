/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: Panda.h 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)
	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
	
 */

/*
#ifdef WIN32 
#include <windows.h> 
#endif 
#include <pthread.h>
*/

#ifndef __PANDA_GLOBAL_H__
#define __PANDA_GLOBAL_H__

#include "cudacpp/Stream.h"

#include "panda/Message.h"
#include "panda/Chunk.h"
#include "panda/PandaMessageIORequest.h"
#include "panda/PandaChunk.h"

#include "panda/PandaMapReduceJob.h"
#include "panda/MapReduceJob.h"
#include "panda/PreLoadedPandaChunk.h"
//
#include "oscpp/AsyncFileReader.h"
#include "oscpp/AsyncIORequest.h"
#include "oscpp/Closure.h"
#include "oscpp/Condition.h"
#include "oscpp/GenericAsyncIORequest.h"
#include "oscpp/Mutex.h"
#include "oscpp/Runnable.h"
#include "oscpp/Thread.h"
#include "oscpp/Win32AsyncIORequest.h"

#endif