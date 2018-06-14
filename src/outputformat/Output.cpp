#include <panda/Output.h>
#include <stdio.h>

namespace panda
{
  Output::Output()
  {
  }
  Output::~Output()
  {
  }
  void Output::write(char *buf, void * key, void *val){
  	printf("Output::write\n");
  }
}
