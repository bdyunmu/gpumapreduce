#include "hsoutputformat.h"
#include <stdio.h>

void HSOutput::write(char *buf, void *key, void *val){
	sprintf(buf,"%s %d\n",(char *)key,*(int *)val);
	return;
}
