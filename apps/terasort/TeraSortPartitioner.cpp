#include <stdio.h>
#include <stdlib.h>
#include "TeraSortPartitioner.h"

byte* TeraSortPartitioner::toByteArray(long value){
	byte* result = new byte[8];
	for(int i = 7;i>=0;i--){
		result[i] = (byte)(value & 0xffL);
		value >>=8;
	}//for
	return result;
}

unsigned long TeraSortPartitioner::toLong(byte * byteArray){
	unsigned long result = (unsigned long)(byteArray[7]);
	long power2 = 1;
	for(int i=6;i>=0;i--){
		power2 *= 256;
		result += (unsigned long)byteArray[i]*power2;
	}
	return result;
}

int TeraSortPartitioner::GetHash(const char *Key, int size, int commSize){
	byte *minbyteArray = new byte[8]{0,0,0,0,0,0,0,0};	
	byte *maxbyteArray = new byte[8]{0,-1,-1,-1,-1,-1,-1,-1};
	unsigned long min = toLong(minbyteArray);
	unsigned long max = toLong(maxbyteArray);
	unsigned long rangePerPart = (max-min)/commSize;
	byte *prefixbyteArray = new byte[8]{0,Key[0],Key[1],Key[2],Key[3],Key[4],Key[5],Key[6]};
	unsigned long prefix = toLong(prefixbyteArray);
	return (prefix/rangePerPart)%commSize;
}

TeraSortPartitioner::TeraSortPartitioner(){

	byte *byteArray = toByteArray(20000);
	for(int i = 0;i<8;i++)
		printf("%d ",(int)byteArray[i]);
	printf("\n");
	long result = toLong(byteArray);
	printf("long:%lu\n",result);
	byte *minbyteArray = new byte[8]{0,0,0,0,0,0,0,0};	
	byte *maxbyteArray = new byte[8]{0,-1,-1,-1,-1,-1,-1,-1};
	for(int i=0;i<8;i++)
		printf("%d ",(unsigned char)maxbyteArray[i]);
	printf("\n");

	unsigned long min = toLong(minbyteArray);
	unsigned long max = toLong(maxbyteArray);
	printf("min:%lu max:%lu\n",min,max);

}

TeraSortPartitioner::~TeraSortPartitioner(){
}
