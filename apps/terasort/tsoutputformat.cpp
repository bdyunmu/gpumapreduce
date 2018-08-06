#include "tsoutputformat.h"
#include <stdio.h>
#include <string.h>

void TSOutput::write(char *buf, void *key, void *val){
	memcpy(buf,key,10);
	memcpy(buf+10,val,90);
	char c = '\n';
	memcpy(buf+100,&c,1);
	//sprintf(buf,"%s%s\n",(char *)key,*(char *)val);
	return;
}
