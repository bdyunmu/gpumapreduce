#include <stdio.h>
#include <stdlib.h>
#include "Long.h"

byte* toByteArray(long value){
byte* result = new byte[8];
for(int i = 7;i>=0;i--){
	result[i] = (byte)(value & 0xffL);
	value >>=8;
}
return result;
}

long toLong(byte * byteArray){
	long result = (long)(byteArray[7]);
	long power2 = 1;
	for(int i = 6;i>=0;i--){
		power2 *= 256;
		result += byteArray[i]*power2;
	}
	return result;
}

int testmain(int argc, char **argv){
	byte *byteArray = toByteArray(20000);
	for(int i = 0;i<8;i++)
		printf("%d ",(int)byteArray[i]);
	printf("\n");
	long result = toLong(byteArray);
	printf("long:%ld\n",result);
	return 0;
}
