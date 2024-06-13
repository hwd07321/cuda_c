#include<stdio.h>
#include<stdlib.h>


// add two vector
__global__ void addvector_GPU(int *g_A, int *g_B, int *g_C){
    int id = threadIdx.x + blockDim.x  * blockIdx.x;
    int value = 0;
    g_C[id] = g_A[id] + g_B[id];
}



// m * k   k * n
__global__ void matmul_GPU(int *g_A, int *g_B, int *g_C, int m, int n, int k){
    int row = threadIdx.y + blcokDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int i = 0;
    int sums = 0;
    for(i ; i < k ; i++){
        sums += g_A[row * m + i] * g_B[i * n + col];      
    }
    g_C[row * m + col] = sums;
}

// reduce sum a vector and this operation is occured in each block
__global__ void reducesum_GPU(int *g_A, int *g_B, int n){
    
    int tid = threadIdx.x;
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if(id >= n){
        return;
    }
    int *data = g_A + blockDim.x * blockIdx.x;
    int stride = 1;
    int index;
    for(stride ; stride < blockDim.x ; stride <<= 1){
        index = 2 * stride * tid;
        if(index < blockDim.x){
            data[index] += data[index+stride];
        }
        __syncthreads();
    }
    if(tid == 0){
        g_B[tid] = data[0];
    }
}   

// scan

__global__ void scannavie_GPU(void *g_A, void *g_B, int n){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int values = 0;
    if(id >= n){
        return ;
    }
    for(int i = 0 ; i < id ; i++){
        values += g_A[i];
    }
    g_B[id] = values;
}

// suposed that the whole data are including in a single block
__global__ void scanH_GPU(int *g_A, int *g_B, int n){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    int values = 0;
    __share__ int data[blockDim.x];
    if(id < n){
        data[id] = g_A[id];
    }
    if(id >= n){
        return ;
    }
    for(int stride = 1 ; stride < n ; stride <<= 1){
        if(id > stride){
            values += data[tid - stride];
        }
        // Becasuse we change the elements in data, so we must make sure that every operation should be sync before.
        __syncthreads;
        data[tid] = values;
        __syncthreads;
    }
    if(id < n){
        g_B[id] = data[id];
    }

}

// quick sorted
// select sorted
// __device__ means that this function only can be called by device
__device__ void select_sorted(int *g_A, int left, int right){
    int min_val = g_A[left];
    int index;
    for(int i = left ; i < right ; i++){
        for(int j = left+1 ; j < right ; j++){
            if(g_A[j] < min_val){
                min_val = g_A[j];
                index = j;
            }
        }
        g_A[index] = g_A[i];
        g_A[i] = min_val;
    }
}

__global__ void Qsort_GPU(int *g_A, int left, int right, int depth){
    if(depth > 32 || right-left <= INTERSECTION){
        select_sorted(g_A, left, right);
    }
    int pivot = g[(left+right)/2];
    int left_ptr = g_A + left;
    int right_ptr = g_A + right;
    while(left <= right){
        while(*left_ptr < pivot){
            left_ptr++;
        }
        while(*right_ptr > pivot){
            right_ptr--;
        }
        if(*left_ptr > *right_ptr){
            int tmp = *left_ptr;
            *left_ptr++ = *right_ptr;
            *right_ptr-- = tmp;
        }
    }
    // find left and right
    int nleft = left_ptr - data;
    int nright = right_ptr - data;

    if(left < nright){
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        // 此时第三个参数0表示动态共享内存大小默认为0，第四个参数是异步stream
        Qsort_GPU<<<1, 1, 0, s>>>(g_A, left, nright, depth+1);
        cudaStreamDestroy(s);
    }
    if(right > nleft){
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        Qsort_GPU<<<1, 1, 0, s>>>(g_A, nleft, right, depth+1);{
            cudaStreamDestroy(s);
        }
    }
}