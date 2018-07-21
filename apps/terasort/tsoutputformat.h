#ifndef __TS_OUTPUT__
#define __TS_OUTPUT__

#include <panda/Output.h>
#include <stdio.h>

class TSOutput: public panda::Output{
	public:
		TSOutput(){
		printf("TSOutput\n");
		}
		~TSOutput(){
		printf("~TSOutput\n");
		}
		void write(char *,void *,void *);
};

#endif
