#ifndef __WC_OUTPUT__
#define __WC_OUTPUT__

#include <panda/Output.h>
#include <stdio.h>

class WCOutput: public panda::Output{
	public:
		WCOutput(){
			printf("WCOutput()\n");
		}
		~WCOutput(){
			printf("~WCOutput()\n");
		}
		void write(char *,void *,void *);
};

#endif
