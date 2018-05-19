#include <mpi.h>

#include <panda/PreLoadedPandaChunk.h>
#include <panda/PandaMessage.h>
#include <panda/PandaMPIMessage.h>
#include <panda/PandaMapReduceJob.h>

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>

int main(int argc, char ** argv)
{

 	if (argc != 2)
        {
	   ShowLog("mpirun -host node1,node2 -np 2 ./%s input.txt",argv[0]);
           exit(-1);//
        }  //if
	if(strlen(argv[1])<2)
	{
	  ShowLog("file path too short!");
	  exit(-1);
	}

	panda::MapReduceJob  *job = new panda::PandaMapReduceJob(argc, argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	job->setMessage(new panda::PandaMPIMessage(true));

	job->setEnableCPU(true);
	job->setEnableGPU(false);

    	char fn[128];
	char str[512];
	sprintf(fn,"%s",argv[1]);
	int  chunk_size = 1024;
	char *chunk_data = (char *)malloc(sizeof(char)*2*(chunk_size));
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
			ShowLog("wordcount job->addInput");
			job->addInput(new panda::PreLoadedPandaChunk((char *)chunk_data, total_len, NUM_ELEMENTS));
			total_len=0;
		}//if
	}//while
	job->execute();
	delete job;
	MPI_Finalize();
	return 0;
}//int main
