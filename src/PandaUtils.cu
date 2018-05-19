/*

        Copyright 2012 The Trustees of Indiana University. All rights reserved.
        Panda: co-processing SPMD computations on GPUs and CPUs.

        File: PandaUtils.cu
        First Version:          2012-07-01 V0.1
        Last UPdates:           2018-04-28 v0.61
        Developer: Hui Li (huili@ruijie.com.cn)
*/

#include "Panda.h"
#include <unistd.h> 
#include <sys/time.h>

#ifndef __PANDA_UTILS_CU__
#define __PANDA_UTILS_CU__

int getGPUCoresNum() { 
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
	//ShowLog("Configure Device ID:%d: Device Name:%s MultProcessorCount:%d sm_per_multiproc:%d", 0, gpu_dev.name,gpu_dev.multiProcessorCount,sm_per_multiproc);
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

double PandaTimer(){
	#ifndef _WIN32
	static struct timeval tv;
	gettimeofday(&tv,NULL);
	double curTime = tv.tv_sec + tv.tv_usec/1000000.0;
	return curTime;
	#else
	double curTime = GetTickCount(); 
	curTime /=1000.0;
	return curTime;
	#endif
}//double PandaTimer()

void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "[PandaError][%s][%i]: CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
		exit((int)err);        
	}
}

#endif //__PANDA_UTILS_CU__
