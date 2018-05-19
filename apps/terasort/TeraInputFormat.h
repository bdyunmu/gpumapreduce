#ifndef _TERA_INPUTFORMAT_H
#define _TERA_INPUTFORMAT_H
#include <stdio.h>
#include <stdlib.h>
#include "Unsigned16.h"
class TeraInputFormat{
public:
	static const int KEY_LEN = 10;
	static const int VALUE_LEN = 90;
	static const int RECORD_LEN = 100;
	static char *inputpath;
	static int recordsPerPartition;
	TeraInputFormat();
	static void copyByte(byte *input, byte *output, int start, int end);
	static void generateRecord(byte *recBuf,Unsigned16 rand,Unsigned16 recordNumber);
};

#endif
