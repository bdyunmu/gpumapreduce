
#include <mpi.h>

#include <panda/PreLoadedPandaChunk.h>
#include <panda/PandaMessage.h>
#include <panda/PandaMPIMessage.h>
#include <panda/PandaMapReduceJob.h>

#include <cudacpp/Event.h>
#include <cudacpp/Stream.h>
#include <oscpp/Timer.h>
#include <string.h>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>

using namespace std;

long sizeStrToBytes(char *str){

	int len = strlen(str);
	int i=0;
	for(i=0; i<len; i++)
		str[i] = tolower(str[i]);
	char lastchar = str[len-1];
	str[len-1] = '\0';	
	long val = 0;

	switch(lastchar){
	case 'k':
	val = atol(str)*1000;	
	break;
	case 'm':
	val = atol(str)*1000*1000;
	break;
	case 'g':
	val = atol(str)*1000*1000*1000;
	break;
	case 't':
	val = atol(str)*1000*1000*1000*1000;
	break;
	default:
	val = val;	
	}
	return val;
}

char *sizeToSizeStr(long sizeInBytes){
	long kbScale = 1000;
	long mbScale = 1000*kbScale;
	long gbScale = 1000*mbScale;
	long tbScale = 1000*gbScale;
	char *sizestr = new char[1024];
	char *p = sizestr;
	if(sizeInBytes > tbScale){
		sprintf(p,"%ld TB",sizeInBytes/tbScale);
	}else if(sizeInBytes > gbScale){
		sprintf(p,"%ld GB",sizeInBytes/gbScale);
	}else if(sizeInBytes > mbScale){
		sprintf(p,"%ld MB",sizeInBytes/mbScale);
	}else if(sizeInBytes > kbScale){
		sprintf(p,"%ld KB",sizeInBytes/kbScale);
	}else {
		sprintf(p,"%ld B",sizeInBytes);
	}
	return sizestr;
}


int main(int argc, char ** argv)
{

 	if (argc != 3)
        {
           ShowLog("terasort");
	   ShowLog("usage: %s [file size][file path]",argv[0]);
	   ShowLog("Example:"); 
	   ShowLog("%s",argv[0]);
	   ShowLog("%s 100G file:///tmp/terasort_in",argv[0]);
           exit(-1);//
        }  //if
	long outputSizeInBytes = sizeStrToBytes(argv[1]);

	panda::MapReduceJob  *job = new panda::PandaMapReduceJob(argc, argv, false,false,true);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	long recordsPerPartition = outputSizeInBytes/100/(long)size;
	long numRecords = recordsPerPartition * size;
	
	job->setMessage(new panda::PandaMPIMessage(true));

	job->setEnableCPU(false);
	job->setEnableGPU(true);

	if (rank == 0)
	{

    	char fn[256];
	char str[512];
	char strInput[1024];
	sprintf(fn,"%s",argv[1]);
	int  chunk_size = 1024;
	ShowLog("start processing txt data...");
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
			job->addInput(new panda::PreLoadedPandaChunk((char *)chunk_data, total_len, NUM_ELEMENTS ));
			total_len=0;
		}//if
	}//while
	ShowLog("rank:[%d] finishing processing txt data",rank);

	}//if

	job->execute();
	//delete job;
	MPI_Finalize();
	return 0;
}//int main
