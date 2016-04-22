#include <mpi.h>
#if 1
#include <panda/PreLoadedPandaChunk.h>
#include <panda/PandaMessage.h>
#include <panda/PandaMapReduceJob.h>

#include <cudacpp/Event.h>
#include <cudacpp/Stream.h>
#include <oscpp/Timer.h>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>

int main(int argc, char ** argv)
{

	panda::MapReduceJob  * job = new panda::PandaMapReduceJob(argc, argv, true);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

//	job->setMessage (new panda::PandaMPIMessage(true));

	//todo here
	if (rank == 0)
	{
    	char fn[256];
	char str[512];
	char strInput[1024];
	sprintf(fn,"sample.txt");
	int  chunk_size = 1024;
	char *chunk_data = (char *)malloc(sizeof(char)*(chunk_size+1000));
	FILE *wcfp;
	wcfp = fopen(fn, "r");

	const int NUM_ELEMENTS = 1;
	int total_len = 0;
	while(fgets(str,sizeof(str),wcfp) != NULL)
	{
		for (int i = 0; i < strlen(str); i++)
		str[i] = toupper(str[i]);
			
		strcpy((chunk_data + total_len),str);
		total_len += (int)strlen(str);
		
		if(total_len>chunk_size){
			job->addInput(new panda::PreLoadedPandaChunk((char *)chunk_data, total_len, NUM_ELEMENTS ));
			job->execute();
			total_len=0;
		}//if
	}//while
	}//if
	delete job;
	return 0;
}//int main
#endif
