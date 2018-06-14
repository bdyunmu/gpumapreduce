#include "wcoutputformat.h"
#include <stdio.h>

void WCOutput::write(char *buf, void *key, void *val){
	sprintf(buf,"%s %d\n",(char *)key,*(int *)val);
	return;
}
