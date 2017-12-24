
#include <mpi.h>

#include <panda/PreLoadedPandaChunk.h>
#include <panda/PandaMessage.h>
#include <panda/PandaMapReduceJob.h>
#include <panda/PandaMapReduceWorker.h>

#include <cudacpp/Event.h>
#include <cudacpp/Stream.h>
#include <oscpp/Timer.h>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>

int main(int argc, char ** argv)
{

 	if (argc != 2)
        {
           ShowLog("word count with panda on cpu and gpu");
           ShowLog("usage: %s [txt path]", argv[0]);
           exit(-1);//
        }  //if
	if(strlen(argv[1])<2)
	{
	ShowLog("txt path too short");
	exit(-1);
	}

	//panda::MapReduceJob  * job = new panda::PandaMapReduceJob(argc, argv, true);
	panda::MapReduceJob  *job = new panda::PandaMapReduceJob(argc, argv, false,false,true);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	job->setMessage(new panda::PandaMPIMessage(true));
	job->setEnableCPU(true);
	//job->setEnableGPU(true);

	//todo here

	//if (rank == 0)
	{
    	char fn[256];
	char str[512];
	char strInput[1024];
	sprintf(fn,argv[1]);
	int  chunk_size = 1024;
	ShowLog("rank:%d, start processing txt data",rank);
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
		if(total_len>=chunk_size){
			ShowLog("word count job->addInput");
			job->addInput(new panda::PreLoadedPandaChunk((char *)chunk_data, total_len, NUM_ELEMENTS ));
			total_len=0;
		}//if
	}//while

	//job->execute();
	ShowLog("rank:%d finishing processing txt data",rank);
	}//if

	job->execute();
	//delete job;
	MPI_Finalize();
	return 0;
}//int main
