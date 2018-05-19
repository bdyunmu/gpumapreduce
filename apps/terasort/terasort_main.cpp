#include <mpi.h>
#include <panda/PreLoadedPandaChunk.h>
#include <panda/PandaMessage.h>
#include <panda/PandaMPIMessage.h>
#include <panda/PandaMapReduceJob.h>

#include <string.h>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>
#include <climits>
#include <assert.h>

#include "TeraSortPartitioner.h"

using namespace std;

int main(int argc, char ** argv)
{

 	if (argc != 3)
        {
           ShowLog("terasort");
	   ShowLog("usage: %s [input][output]",argv[0]);
	   ShowLog("Example:");
	   ShowLog("mpirun -host node1,node2 -np 2 ./%s file:///tmp/terasort_in file:///tmp/terasort_out",argv[0]);
           exit(-1);//
        }  //if
	panda::MapReduceJob  *job = new panda::PandaMapReduceJob(argc, argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	job->setPartition(new TeraSortPartitioner());	
	job->setMessage(new panda::PandaMPIMessage(true));

	job->setEnableCPU(true);
	job->setEnableGPU(false);

	if (rank == 0)
	{
	ShowLog("========================================================");
	ShowLog("========================================================");
	ShowLog("TeraSort");
	ShowLog("Input:%s",argv[1]);
	ShowLog("Output:%s",argv[2]);
	ShowLog("=========================================================");
	ShowLog("=========================================================");	
	}
	
	const int NUM_ELEMENTS = 1;
	char input[64];
	sprintf(input,"%s/INPUT%d",argv[1],rank);
	FILE *fp = fopen(input,"r");
	if(fp == NULL){
	ShowLog("can not open:%s",input);
	exit(-1);
	}
	char rb[100];
	int count = 0;
	while(fread(rb,100,1,fp)!=NULL){
	job->addInput(new panda::PreLoadedPandaChunk((char *)rb,100,NUM_ELEMENTS));
	count++;	
	}
	ShowLog("terasort job->addInput count:[%d]",count);
	fclose(fp);
	job->execute();
	delete job;
	MPI_Finalize();
	return 0;
}//int main
