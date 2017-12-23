
/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: PandaUtils.cu 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02
	Developer: Hui Li (lihui@indiana.edu)
	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */


#include "Panda.h"

#ifdef _WIN32 
#include <windows.h> 
#include <time.h>
#elif MACOS 
#include <sys/param.h> 
#include <sys/sysctl.h> 
#elif __linux
#include <unistd.h> 
#include <sys/time.h>
#endif 

#ifndef __PANDAUTILS_CU__
#define __PANDAUTILS_CU__


int getGPUCoresNum() { 
	//assert(tid<total);
	int arch_cores_sm[3] = {1, 8, 32 };
	cudaDeviceProp gpu_dev;
	int tid = 0;
	cudaGetDeviceProperties(&gpu_dev, tid);

	int sm_per_multiproc = 1;
	if (gpu_dev.major == 9999 && gpu_dev.minor == 9999)
			sm_per_multiproc = 1;
	else if (gpu_dev.major <=2)
			sm_per_multiproc = arch_cores_sm[gpu_dev.major];
	else
			sm_per_multiproc = arch_cores_sm[2];

	//return ((gpu_dev.multiProcessorCount)*(sm_per_multiproc));
	ShowLog("Configure Device ID:%d: Device Name:%s MultProcessorCount:%d sm_per_multiproc:%d", 0, gpu_dev.name,gpu_dev.multiProcessorCount,sm_per_multiproc);
	return ((gpu_dev.multiProcessorCount)*(sm_per_multiproc));
}

void sleep(int sleepMs)
{
#ifdef __linux
    usleep(sleepMs * 1000);   // usleep takes sleep time in us
#endif
#ifdef _WIN32
    Sleep(sleepMs);
#endif
}



int getCPUCoresNum() { 

#ifdef WIN32 
    SYSTEM_INFO sysinfo; 
    GetSystemInfo(&sysinfo); 
    return sysinfo.dwNumberOfProcessors; 
#elif MACOS 
    int nm[2]; 
    size_t len = 4; 
    uint32_t count; 
 
    nm[0] = CTL_HW; nm[1] = HW_AVAILCPU; 
    sysctl(nm, 2, &count, &len, NULL, 0); 
 
    if(count < 1) { 
        nm[1] = HW_NCPU; 
        sysctl(nm, 2, &count, &len, NULL, 0); 
        if(count < 1) { count = 1; } 
    } 
    return count; 
#elif __linux
    return sysconf(_SC_NPROCESSORS_ONLN); 
#endif 

}


void DoDiskLog(const char *str){
	FILE *fptr;
	char file_name[128];
	sprintf(file_name,"%s","panda.log");
	fptr = fopen(file_name,"a");
	fprintf(fptr,"[PandaDiskLog]\t\t:%s\n",str);
	//fprintf(fptr,"%s",__VA_ARGS__);
	fclose(fptr);
	//printf("\n");
}//void

double PandaTimer(){

	#ifndef _WIN32
	static struct timeval tv;
	gettimeofday(&tv,NULL);
	double curTime = tv.tv_sec + tv.tv_usec/1000000.0;

	//ShowLog("\t Panda CurTime:%f", curTime);
	return curTime;
	#else
	//newtime = localtime( &long_time2 ); 
	double curTime = GetTickCount(); 
	//ShowLog("\t Panda CurTime:%f", curTime);
	curTime /=1000.0;
	return curTime;
	#endif

}

void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "[PandaError][%s][%i]: CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
		exit((int)err);        
	}
}


//--------------------------------------------------------
//start_task_id a timer
//param	: start_row_id_tv
//--------------------------------------------------------

//--------------------------------------------------------
//end a timer, and print out a message
//--------------------------------------------------------

#endif //__PANDAUTILS_CU__
