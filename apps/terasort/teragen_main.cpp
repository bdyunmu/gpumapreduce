
#include <mpi.h>

#include <panda/DataChunk.h>
#include <panda/PandaMPIMessage.h>
#include <panda/PandaMapReduceJob.h>

#include <string.h>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctype.h>
#include <climits>
#include <assert.h>

#include "TeraInputFormat.h"

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
           panda::ShowLog("teragen");
	   panda::ShowLog("usage: %s [file size][file path]",argv[0]);
	   panda::ShowLog("Example:"); 
	   panda::ShowLog("mpirun -host node1,node2 -np 2 ./%s 100G file:///tmp/terasort_in",argv[0]);
           exit(-1);//
        }  //if
	long outputSizeInBytes = sizeStrToBytes(argv[1]);
	char *sizeStr = sizeToSizeStr(outputSizeInBytes);

	panda::PandaMapReduceJob  *job = new panda::PandaMapReduceJob(argc, argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	long recordsPerPartition = outputSizeInBytes/100/(long)size;
	long numRecords = recordsPerPartition * size;
	assert(recordsPerPartition < INT_MAX);
	
	job->setMessage(new panda::PandaMPIMessage(true));

	job->setEnableCPU(true);
	job->setEnableGPU(false);

	if (rank == 0)
	{
	panda::ShowLog("========================================================");
	panda::ShowLog("========================================================");
	panda::ShowLog("Input Size:%s",sizeStr);
	panda::ShowLog("Total number of records:%ld",numRecords);
	panda::ShowLog("Number of output partitions:%d",size);
	panda::ShowLog("Number of records/output parititon:%d",numRecords/size);
	panda::ShowLog("=========================================================");
	panda::ShowLog("=========================================================");	
	}

	TeraInputFormat::recordsPerPartition = recordsPerPartition;
	TeraInputFormat::inputpath = argv[2];
	
	const int NUM_ELEMENTS = 1;
	int *index = new int[1];
	*index = rank;
	job->addInput(new panda::DataChunk((char *)index,sizeof(int),NUM_ELEMENTS));
	panda::ShowLog("teragen job->addInput index:[%d]",*index);

	job->execute();
	delete job;
	//MPI_Finalize();
	return 0;
}//int main
