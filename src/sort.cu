#include <panda/Sorter.h>
#include <cuda.h>

namespace panda
{
  Sorter::Sorter(){
  }
  Sorter::~Sorter(){}
 
 int Sorter::cpu_compare(const void *key_a,int len_a, const void *key_b,int len_b){
	int short_len = len_a>len_b? len_b:len_a;
        for(int i = 0;i<short_len;i++){
                if(((char *)key_a)[i]>((char *)key_b)[i])
                return 1;
                if(((char *)key_a)[i]<((char *)key_b)[i])
                return -1;
        }
        return 0;
  }

  int __device__ Sorter::gpu_compare(const void *key_a, int len_a, const void *key_b, int len_b){

    int short_len = len_a>len_b? len_b:len_a;
    for(int i = 0;i<short_len;i++){
      	if(((char *)key_a)[i]>((char *)key_b)[i])
        return 1;
      	if(((char *)key_a)[i]<((char *)key_b)[i])
        return -1;
    }
    return 0;

  }
  bool Sorter::canExecuteOnGPU(){
   	return false;
  } 
  bool Sorter::canExecuteOnCPU() {
	return false;
  }
  void Sorter::init() {
  }
  void Sorter::finalize() {
  }
  void Sorter::executeOnGPUAsync(void * const keys, void * const vals, const int numKeys, int & numUniqueKeys, int ** keyOffsets, int ** valOffsets, int ** numVals){}
  void Sorter::executeOnCPUAsync(void * const keys, void * const vals, const int numKeys, int & numUniqueKeys, int ** keyOffsets, int ** valOffsets, int ** numVals){}

}

