/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	File: reduce.cu 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)
	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */

#ifndef __USER_CU__
#define __USER_CU__

#include "Panda.h"
#include "UserAPI.h"

//-------------------------------------------------------------------------
//Reduce Function in this application
//-------------------------------------------------------------------------

__device__ int gpu_compare(const void *d_a, int len_a, const void *d_b, int len_b)
{
	return 0;
}

__device__ void gpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context *d_g_state, int map_task_idx)
{
	return;
}

__device__ void gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context d_g_state){

		return;
		
}//reduce2

__device__ float operator*(float4 a, float4 b)
{
	return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
}//__device__


int cpu_compare(const void *d_a, int len_a, const void *d_b, int len_b)
{
	return 0;
}


void cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context *d_g_state, int map_task_idx)
{
	return;
}



void cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context* d_g_state){
		return;
}//reduce2



void cpu_1d_blocked_matrix(float *A, float *B, float *C, int wA,int start_task_id,int end_id, int bz);
void cpu_2d_blocked_matrix(float *A, float *B, float *C, int wA,int row_id,int col_id, int bz);

void cpu_map(void *KEY, void*VAL, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){

	MM_KEY_T* pKey = ((MM_KEY_T*)KEY);
	MM_VAL_T* pVal = ((MM_VAL_T*)VAL);

	int rowId = pVal->row;
	int colId = pVal->col;
	int bz = MATRIX_BLOCK_SIZE;

	int wA = pVal->col_dim;
	int wB = pVal->col_dim;

	float *A = pKey->h_matrix1;
	float *B = pKey->h_matrix2;
	float *C = pKey->h_matrix3;
	cpu_1d_blocked_matrix(A, B, C, wA,rowId,colId,bz);
	
}//map2

void gpu_card_map(void *key, void *val, int keySize, int valSize, gpu_card_context *d_g_state, int map_task_idx){

}//void

int gpu_card_compare(const void *d_a, int len_a, const void *d_b, int len_b){
	return 0;
}//int

//Last update 9/2/2012
//blocked matrix useful
__device__ void gpu_map1(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	MM_KEY_T* pKey = ((MM_KEY_T*)KEY);
	MM_VAL_T* pVal = ((MM_VAL_T*)VAL);


	int wA = pVal->col_dim;
	int wB = pVal->col_dim;
	
	int m = wA;
	int bz = MATRIX_BLOCK_SIZE;

	float Csub = 0.0;
	float *A = pKey->matrix1;
	float *B = pKey->matrix2;
	float *C = pKey->matrix3;

	float4*As = (float4*)A;
	float4*Bs = (float4*)B;

	int i,j,k;
	int start_row_id_a_matrix = pVal->row*bz;
	int start_row_id_b_matrix = pVal->col*bz;

	int aHeight = bz;
	int aHeightBlocks = aHeight/bz;
	int aLastBlockHeight = aHeight - (aHeightBlocks*bz);

	if (aLastBlockHeight>0){
		aHeightBlocks++;
	}//if

	int bWidth = bz;
	int bWidthBlocks = bWidth/bz;
	int bLastBlockWidth = bWidth - (bWidthBlocks*bz);
	if (bLastBlockWidth>0){
		bWidthBlocks++;
	}//if

	int commBlocks = m/bz;
	int commLastBlockWidth = m - (commBlocks*bz);
	if (commLastBlockWidth >0){
		commBlocks++;
	}//fi

	int aBlockHeight = bz;
	int bBlockWidth = bz;
	int commBlockWidth = bz;
	int ib,jb,kb;
	float4 b4,c4;
	float aik;

	for (ib=0; ib<aHeightBlocks; ib++){
		if (aLastBlockHeight>0 && ib==(aHeightBlocks-1)){
			aBlockHeight = aLastBlockHeight;
		}//if

		bBlockWidth = bz;
		for (jb=0; jb<bWidthBlocks;jb++){
			if (bLastBlockWidth>0&&jb==(bWidthBlocks-1))
				bBlockWidth = bLastBlockWidth;

			commBlockWidth = bz;
			for (kb =0;kb<commBlocks;kb++){
				if (commLastBlockWidth>0 && kb==(commBlocks-1))
					commBlockWidth = commLastBlockWidth;

				for (i = start_row_id_a_matrix + ib*bz;i<start_row_id_a_matrix+(ib*bz)+aBlockHeight;i++){
					for (k = kb*bz;k<(kb*bz)+(commBlockWidth);k++){
						aik = A[i*m+k];
						float4 *Bsub = (float4*)(B+k*m+jb*bz);
						float4 *Csub = (float4*)(C+i*m+jb*bz);
						//for (j= jb*bz;j<(jb*bz)+(bBlockWidth)/4;j++){
						for (j=0; j<(bBlockWidth/4); j++){
							b4 = *((Bsub)+j);
							c4 = *((Csub)+j);
							c4.x += aik*b4.x;
							c4.y += aik*b4.y;
							c4.z += aik*b4.z;
							c4.w += aik*b4.w;
							*((Csub)+j) = c4;
							//(C[i*m+j]+=A[i*m+k]*B[k*m+j];
						}//for
						int indexBase = jb*bz+4*(bBlockWidth/4);
						for (int rj=0; rj<(bBlockWidth%4); rj++){
							int index = indexBase + rj;
							C[i*m+index] += aik*(*(B+ k*m +index));
						}
					}
				}//for
			}//for
		}//for
	}//for
	//check results	
	/*if (map_task_idx == 1){
		for (int j=10;j<20;j++)
		for (int i=0;i<5;i++){
			printf("%f ",C[j*wA+i]);
		}
		printf("\n");
	}*/
}

//Last Update 9/24/2012
__device__ void gpu_core_map(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	MM_KEY_T* pKey = ((MM_KEY_T*)KEY);
	MM_VAL_T* pVal = ((MM_VAL_T*)VAL);
	
	int wA = pVal->col_dim;
	int wB = pVal->col_dim;
	//int tbz = pVal->tbz; 
	//int mbz = pVal->mbz;
	int tbz = THREAD_BLOCK_SIZE;
	int mbz = MATRIX_BLOCK_SIZE;
	int m = wA;
	
	

	float Csub = 0.0;
	float *A = pKey->matrix1;
	float *B = pKey->matrix2;
	float *C = pKey->matrix3;
	
	int start_row_id_a_matrix = pVal->row*mbz;
	int start_row_id_b_matrix = pVal->col*mbz;

	int aHeight = mbz;
	int aHeightBlocks = aHeight/tbz;
	int aLastBlockHeight = aHeight - (aHeightBlocks*tbz);
	if (aLastBlockHeight>0){
		//aHeightBlocks++;
	}//if

	int bWidth = mbz;
	int bWidthBlocks = bWidth/tbz;
	int bLastBlockWidth = bWidth - (bWidthBlocks*tbz);
	if (bLastBlockWidth>0){
		//bWidthBlocks++;
	}//if

	int commBlocks = m/tbz;
	int commLastBlockWidth = m - (commBlocks*tbz);
	if (commLastBlockWidth >0){
		//commBlocks++;
	}//fi
	int aBlockHeight = tbz;
	int bBlockWidth = tbz;
	int commBlockWidth = tbz;
	int ib,jb,kb;	

	//int bx = blockIdx.x;
	//int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	__shared__ float As[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];
	__shared__ float Bs[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];
	__shared__ int row_id_a;
	__shared__ int row_id_b;

	//printf("wA:%d wB:%d tx:%d ty:%d\n",wA,wB,tx,ty);

	for (ib=0; ib<aHeightBlocks; ib++){
		if (aLastBlockHeight>0 && ib==(aHeightBlocks-1)){
			aBlockHeight = aLastBlockHeight;
		}//if

		bBlockWidth = tbz;
		for (jb=0; jb<bWidthBlocks;jb++){
			if (bLastBlockWidth>0 && jb==(bWidthBlocks-1)){
				bBlockWidth = bLastBlockWidth;
			}

			/*commBlockWidth = tbz;
			for (kb =0;kb<commBlocks;kb++){
				if (commLastBlockWidth>0 && kb==(commBlocks-1))
					commBlockWidth = commLastBlockWidth;*/

			for (int y=0;y<THREAD_BLOCK_SIZE;y++){
				for (int x=0;x<THREAD_BLOCK_SIZE;x++){
				Csub = 0.0;					
				if (y*THREAD_BLOCK_SIZE+x==THREAD_ID){
					row_id_a = start_row_id_a_matrix+ib*tbz;
					row_id_b = start_row_id_b_matrix+jb*tbz;
				}//if	
					
				__syncthreads();
							
				int row_id = (row_id_a + ty);
				if (row_id >= m) row_id = m-1;
				int col_id = (row_id_b + tx);
				if (col_id >= m) col_id = m-1;
		
				for (int cb=0; cb<commBlocks; cb++){
				
				As[ty][tx] = A[(row_id)*m + cb*tbz + tx];
				Bs[ty][tx] = B[(col_id)*m + cb*tbz + tx];
				
				//if(cb==commBlocks-1)
				//	printf("row:%d col:%d  index:%d As[%d][%d]:%f\n",row_id,col_id,(row_id)*m + cb*tbz + tx,ty,tx,As[ty][tx]);
				
				__syncthreads();
				
				#pragma unroll		
					for (int k = 0; k<THREAD_BLOCK_SIZE; k++)
						Csub += As[ty][k]*Bs[k][tx];
				__syncthreads();
				
				}//for
				
				int index = (row_id)*m + (col_id);
				if(index>m*m-1){
					printf("error! index>m*m-1\n");
					index = m*m-1;
				}

				C[index] = Csub;
				printf("Csub:%f\n",Csub);

				}//for (int x=0;x<16;x++)
			}//for (int y=0;y<16;y++)
		}//for (jb=0
	}
}


//Last Updated 9/1/2012
//CUDA implementation of Matrix Multiplication useful
__device__ void gpu_map2(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	//printf("map_task_idx:%d\n",map_task_idx);
	
	MM_KEY_T* pKey = ((MM_KEY_T*)KEY);
	MM_VAL_T* pVal = ((MM_VAL_T*)VAL);

	int wA = pVal->col_dim;
	int wB = pVal->col_dim;
	//int bz = pVal->tbz; //size of each tile
	int bz = MATRIX_BLOCK_SIZE;
	int m = wA;

	float Csub = 0.0;
	float *A = pKey->matrix1;
	float *B = pKey->matrix2;
	float *C = pKey->matrix3;
	
	int start_row_id_a_matrix = pVal->row*bz;
	int start_row_id_b_matrix = pVal->col*bz;

	int aHeight = bz;
	int aHeightBlocks = aHeight/bz;
	int aLastBlockHeight = aHeight - (aHeightBlocks*bz);
	if (aLastBlockHeight>0){
		aHeightBlocks++;
	}//if
	int bWidth = bz;
	int bWidthBlocks = bWidth/bz;
	int bLastBlockWidth = bWidth - (bWidthBlocks*bz);
	if (bLastBlockWidth>0){
		bWidthBlocks++;
	}//if
	int commBlocks = m/bz;
	int commLastBlockWidth = m - (commBlocks*bz);
	if (commLastBlockWidth >0){
		//commBlocks++;
	}//fi
	int aBlockHeight = bz;
	int bBlockWidth = bz;
	int commBlockWidth = bz;
	int ib,jb,kb;	

	//int bx = blockIdx.x;
	//int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	__shared__ float As[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];
	__shared__ float Bs[THREAD_BLOCK_SIZE][THREAD_BLOCK_SIZE];
	__shared__ int row_id_a;
	__shared__ int row_id_b;
	

	for (int y=0;y<MATRIX_BLOCK_SIZE;y++){
		for (int x=0;x<MATRIX_BLOCK_SIZE;x++){
			Csub = 0.0;					
			if (y*MATRIX_BLOCK_SIZE+x==THREAD_ID){
				row_id_a = start_row_id_a_matrix;
				row_id_b = start_row_id_b_matrix;
			}//if	
					
			__syncthreads();

			int row_id = (row_id_a+ty);
			if (row_id >= m) row_id = m-1;
			int col_id = (row_id_b+tx);
			if (col_id >= m) col_id = m-1;
			


			for (int cb=0; cb<commBlocks; cb++){
			
				//As[ty][tx] = A[(row_id_a + ty)*m + cb*bz + tx];
				//Bs[ty][tx] = B[(row_id_b + ty)*m + cb*bz + tx];
				As[ty][tx] = A[(row_id)*m + cb*bz + tx];
				Bs[ty][tx] = B[(row_id)*m + cb*bz + tx];

				__syncthreads();
				#pragma unroll		
					for (int k = 0; k < MATRIX_BLOCK_SIZE; k++)
						Csub += As[ty][k]*Bs[k][tx];
				__syncthreads();
			}//for

			//if ((x==0) && (y==5)&&(map_task_idx%50==1))
			//	printf("commBlocks:%d map_task_idx:%d tx:%d ty:%d Csub:%f index:%d row_id_a:%d bz:%d THREAD_ID:%d\n",commBlocks, map_task_idx, tx, ty, Csub, index, row_id_a, bz, THREAD_ID);
			int index = (row_id)*m + (col_id);
			C[index] = Csub;
			
		}//for
	}
}


#endif //__USER_CU__