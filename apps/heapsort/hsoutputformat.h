#ifndef __HS_OUTPUT__
#define __HS_OUTPUT__

#include <panda/Output.h>
#include <stdio.h>

class HSOutput: public panda::Output{
	public:
		HSOutput(){
			printf("HSOutput()\n");
		}
		~HSOutput(){
			printf("~HSOutput()\n");
		}
		void write(char *,void *,void *);
};

#endif
