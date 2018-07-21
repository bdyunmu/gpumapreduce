#include "tsoutputformat.h"
#include <stdio.h>

void TSOutput::write(char *buf, void *key, void *val){
	sprintf(buf,"%s%s\n",(char *)key,*(char *)val);
	return;
}
