#include <mpi.h>
#include "Panda.h"

#include <panda/KeyValueChunk.h>
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
#include "TeraInputFormat.h"
#include "tsoutputformat.h"

using namespace std;

int main(int argc, char ** argv)
{
	
 	if (argc != 3)
        {
           panda::ShowLog("terasort");
	   panda::ShowLog("usage: %s [input][output]",argv[0]);
	   panda::ShowLog("Example:");
	   panda::ShowLog("mpirun -host node1,node2 -np 2 ./%s file:///tmp/terasort_in file:///tmp/terasort_out",argv[0]);
           exit(-1);
        }//if
	panda::PandaMapReduceJob  *job = new panda::PandaMapReduceJob(argc, argv);
	int rank, size;
	rank = 0;
	size = 1;
	//MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//MPI_Comm_size(MPI_COMM_WORLD, &size);
	job->setOutput(new TSOutput());
	job->setPartition(new TeraSortPartitioner());	
	job->setMessage(new panda::PandaMPIMessage(true));

	job->setEnableCPU(true);
	job->setEnableGPU(false);

	if (rank == 0)
	{
	panda::ShowLog("========================================================");
	panda::ShowLog("========================================================");
	panda::ShowLog("TeraSort");
	panda::ShowLog("Input:%s",argv[1]);
	panda::ShowLog("Output:%s",argv[2]);
	panda::ShowLog("=========================================================");
	panda::ShowLog("=========================================================");	
	}
	
	char input[128];
	sprintf(input,"%s/INPUT%d",argv[1],rank);
	FILE *fp = fopen(input,"r");
	if(fp == NULL){
		panda::ShowLog("can not open:%s",input);
		exit(-1);
	}
	char *rb  = (char *)malloc(sizeof(char)*100);
	int count = 0;
	while(fread(rb,100,1,fp)!=0){
		if(count == 8)
			break;
		printf("read key:");
		for(int s = 0;s<10;s++){
		printf("%3d",(int)rb[s]);
		}
		printf("\n");
		job->addInput(new panda::KeyValueChunk((char *)rb,TeraInputFormat::KEY_LEN,(char *)rb+10,TeraInputFormat::VALUE_LEN));
		count++;
		rb = (char *)malloc(sizeof(char)*100);
	}
	panda::ShowLog("terasort keyvalue pairs num:%d",count);
	fclose(fp);
	job->execute();
	//delete job;
	return 0;
}//int main
