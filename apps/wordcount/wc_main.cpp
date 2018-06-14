#include <mpi.h>

#include <panda/DataChunk.h>
#include <panda/PandaMPIMessage.h>
#include <panda/PandaMapReduceJob.h>
#include "wcoutputformat.h"
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>

int main(int argc, char ** argv)
{
 	if (argc != 2)
        {
	   panda::ShowLog("mpirun -host node1,node2 -np 2 ./%s input.txt",argv[0]);
           exit(-1);//
        }  //if
	if(strlen(argv[1])<2)
	{
	  panda::ShowLog("file path too short!");
	  exit(-1);
	}
	panda::PandaMapReduceJob  *job = new panda::PandaMapReduceJob(argc, argv);
	job->setOutput(new WCOutput());
	job->setMessage(new panda::PandaMPIMessage(true));
	job->setTaskLevel(panda::TaskLevelTwo);
	job->setEnableCPU(true);
	job->setEnableGPU(true);

    	char wcfn[128];
	char str[1024];
	sprintf(wcfn,"%s",argv[1]);
	int  chunk_size = 1024;
	char *chunk_data = (char *)malloc(sizeof(char)*2*(chunk_size));
	FILE *wcfp;
	wcfp = fopen(wcfn, "r");
	const int NUM_ELEMENTS = 1;
	int total_len = 0;
	while(fgets(str,sizeof(str),wcfp) != NULL)
	{
		for (int i = 0; i < strlen(str); i++)
		str[i] = toupper(str[i]);
		strcpy((chunk_data + total_len),str);
		total_len += (int)strlen(str);
		if(total_len>=chunk_size){
			panda::ShowLog("(wordcount job->addInput)");
			job->addInput(new panda::DataChunk((char *)chunk_data, total_len, NUM_ELEMENTS));
			total_len=0;
		}//if
	}//while
	if(total_len >0){
		panda::ShowLog("(wordcount job->addInput)");
		job->addInput(new panda::DataChunk((char *)chunk_data, total_len, NUM_ELEMENTS));
	}
	panda::ShowLog("(wordcount job->execute)");
	job->execute();
	delete job;
	return 0;

}//int main
