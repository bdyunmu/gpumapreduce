#ifndef __PANDA_MESSAGE_PACKAGE_H__
#define __PANDA_MESSAGE_PACKAGE_H__
#include <oscpp/Condition.h>
#include <oscpp/Mutex.h>

namespace panda{
class PandaMessagePackage{

public:
	void *keys;
	void *vals;
	int keySize;
	int valSize;

	void * keysBuff;
	int keyBuffSize;
	void * valsBuff;
	int valBuffSize;
	int *keyPosKeySizeValPosValSize;
	int *counts;
	int rank;
	MPI_Status *stats;
	MPI_Request *reqs;
	volatile bool *flag;
	volatile bool *waiting;
	bool done[4];
	oscpp::Condition *cond;	
};
}

#endif
