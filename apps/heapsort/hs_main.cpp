#include <mpi.h>

#include <panda/DataChunk.h>
#include <panda/PandaMPIMessage.h>
#include <panda/PandaMapReduceJob.h>
#include "hsoutputformat.h"
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>

int main(int argc, char ** argv)
{
 	if (argc != 2)
        {
	   panda::ShowLog("mpirun -host node1,node2 -np 2 ./%s input.txt",argv[0]);
           exit(-1);
        }  //
	if(strlen(argv[1])<2)
	{
	  panda::ShowLog("file path too short!");
	  exit(-1);
	}
	panda::PandaMapReduceJob  *job = new panda::PandaMapReduceJob(argc, argv);
	job->setOutput(new HSOutput());
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
	int total_len = 0;
	int chunk_count = 0;
	while(fgets(str,sizeof(str),wcfp) != NULL)
	{
		for (int i = 0; i < strlen(str); i++){
			str[i] = toupper(str[i]);
		}
		strncpy((chunk_data + total_len),str,strlen(str));
		total_len += (int)strlen(str);
		if(total_len>=chunk_size){
			panda::ShowLog("(heapsort job->addInput) chunk_count:%d",chunk_count);
			job->addInput(new panda::DataChunk(chunk_count,(char *)chunk_data, total_len));
			total_len=0;
			chunk_count++;
		}//if
	}//while
	if(total_len >0){
		panda::ShowLog("(heapsort job->addInput) chunk_count:%d",chunk_count);
		job->addInput(new panda::DataChunk(chunk_count,(char *)chunk_data, total_len));
	}
	panda::ShowLog("(heapsort job->execute)");
	job->execute();
	delete job;
	return 0;

}//int main
