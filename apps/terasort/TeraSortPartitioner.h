#ifndef _TERASORT_PARTITIONER_H_
#define _TERASORT_PARTITIONER_H_
#include <panda/Partitioner.h>
typedef char byte;

class TeraSortPartitioner: public panda::Partitioner{
public:
	TeraSortPartitioner();
	~TeraSortPartitioner();
	int GetHash(const char *Key, int KeySize, int commSize);
private:
	byte* toByteArray(long value);
	unsigned long toLong(byte * byteArray);
};

#endif
