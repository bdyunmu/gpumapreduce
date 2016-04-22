#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
//#include "MTRand.h"
#if 0
int main2(int argc, char ** argv)
{
  int numElems, numDims, seed;
  float * elems;
  if (argc != 5)
  {
    printf("Usage: %s output_file num_elements num_dimensions random_seed\n", *argv);
    return 0;
  }//if

  FILE * fp = fopen(argv[1], "wb");
  seed = atoi(argv[4]);
  MTRand mtrand(seed);

  numDims = atoi(argv[3]);
  elems = new float[16 * 1024 * numDims];
  numElems = atoi(argv[2]);
  int numLeft = numElems;
  printf("writing %s.\n", argv[1]); fflush(stdout);
  while (numLeft > 0)
  {
    printf("\r                                                        \r%d / %d", numElems - numLeft, numElems); fflush(stdout);
    int numToGen = std::min(16 * 1024, numLeft);
    numLeft -= numToGen;
    for (int i = 0; i < numToGen * numDims; ++i)
    {
      do
      {
        elems[i] = static_cast<float>(mtrand.rand());
      }//do
      //while (isnan(elems[i]));
	  while ((elems[i])!=NULL);
    }
    if (fwrite(elems, sizeof(float) * numToGen * numDims, 1, fp) != 1)
    {
      printf("error writing.\n");
      fflush(stdout);
    }
  }
  printf("\r                                                          \r%d / %d\ndone\n", numElems, numElems); fflush(stdout);

  fclose(fp);
}
#endif
