/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	Panda:a MapReduce Framework on GPUs and CPUs
	
	File: PandaSched.cu 
	First Version:		2012-07-01 V0.1
	Last Updates:		2018-05-22 V0.43

	Developer: Hui Li (huili@ruijie.com.cn)

 */

#include "Panda.h"
namespace panda{

//calculate the workload ratio between gpu and cpu using the roufline model.
void pandaTaskSched(panda_node_context *pnc, panda_gpu_context *pgc, panda_cpu_context *pcc){
	int gpuTotalGHz = pgc->num_gpus_cores*pgc->gpu_GHz;
	int cpuTotalGHz = pcc->num_cpus_cores*pcc->cpu_GHz;
	pnc->cpu_ratio = cpuTotalGHz/(gpuTotalGHz+cpuTotalGHz);
	pnc->gpu_ratio = 1- pnc->cpu_ratio;	
}

}
