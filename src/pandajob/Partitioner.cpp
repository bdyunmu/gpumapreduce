#include <panda/Partitioner.h>

namespace panda
{
	Partitioner::Partitioner(){}
	Partitioner::~Partitioner(){}
	int Partitioner::GetHash(const char *Key, int KeySize, int commSize){
	/////FROM : http://courses.cs.vt.edu/~cs2604/spring02/Projects/4/elfhash.cpp
        unsigned long h = 0;
        while(KeySize-- > 0)
        {
                h = (h << 4) + *Key++;
                unsigned long g = h & 0xF0000000L;
                if (g) h ^= g >> 24;
                h &= ~g;
        }//while
        return (int) ((int)h % commSize);
	}

}
