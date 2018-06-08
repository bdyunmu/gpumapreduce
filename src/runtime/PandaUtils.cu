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
#include <memory>
#include <stdio.h>
#ifndef __PANDA_UTILS_CU__
#define __PANDA_UTILS_CU__


void getGPUDevProp(){
	int devCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&devCount);
	if(error_id != cudaSuccess){
	printf("Result = FAIL\n");
	exit(-1);
	}
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp gpu_dev;
	cudaGetDeviceProperties(&gpu_dev,dev);
	printf("Total amount of global memory: %.0f Mbytes\n",(float)gpu_dev.totalGlobalMem/1048576.0f);
	printf("GPU Max Clock rate: %.0f GHz\n",gpu_dev.clockRate*1e-6f);
	printf(" Memory Clock rate:  %.0f Mhz\n",gpu_dev.memoryClockRate*1e-3f);
	printf(" Memory Bus Width:  %d-bit\n",gpu_dev.memoryBusWidth);
}
int getGPUMemBandwidth(){
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp gpu_dev;
	cudaGetDeviceProperties(&gpu_dev,dev);
	int gmbd = (gpu_dev.memoryClockRate*1e-6f)*(gpu_dev.memoryBusWidth)/8*2;
	return gmbd; //in GB/s
}
int getGPUMemSize(){
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp gpu_dev;
	cudaGetDeviceProperties(&gpu_dev,dev);
	return (int)(gpu_dev.totalGlobalMem/1048576.0f);
}
int getGPUGHz(){
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp gpu_dev;
	cudaGetDeviceProperties(&gpu_dev,dev);
	return (int)(gpu_dev.clockRate*1e-6f);
}
int getGPUCoresNum() { 
	int arch_cores_sm[3] = {1, 8, 32 };
	cudaDeviceProp gpu_dev;
	int tid = 0;
	cudaGetDeviceProperties(&gpu_dev,tid);
	int sm_per_multiproc = 1;
	if (gpu_dev.major == 9999 && gpu_dev.minor == 9999)
			sm_per_multiproc = 1;
	else if (gpu_dev.major <=2)
			sm_per_multiproc = arch_cores_sm[gpu_dev.major];
	else
			sm_per_multiproc = arch_cores_sm[2];
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
int getCPUMemSize(){
	char cmd[128];
	sprintf(cmd,"cat /proc/meminfo |grep MemTotal|awk -F:' ' '{print $2}'");
	FILE *fp = popen(cmd,"r");
	if(fp == NULL){
		printf("cat /proc/meminfo/ fp ==  NULL\n");
		exit(0);
	}//if
	char buf1[128];
	fread(buf1,128,1,fp);
	pclose(fp);
	int cpuMemSize;
	char buf2[128];
	sscanf(buf1,"%d %s",&cpuMemSize,buf2);
	return cpuMemSize;
}

int getCPUMemBandwidth(){
	return 0;
}

double getCPUGHz(){
	char cmd[128];
	sprintf(cmd,"cat /proc/cpuinfo |grep MHz|awk -F':' '{print $2}'|head -n 1");
	FILE *fp = popen(cmd,"r");
	if(fp == NULL){
		printf("cat /proc/cpuinfo/ fp == NULL\n");
		exit(0);
	}
	char buf[128];
	fread(buf,128,1,fp);
	pclose(fp);
	double CPUGHz = 0;
	sscanf(buf,"%lf",&CPUGHz);
	return CPUGHz/1000.0;
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
