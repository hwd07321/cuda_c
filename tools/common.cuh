#include<stdio.h>
#include<stdlib.h>

cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess)
    {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n",
                error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}
// cudaError_t error = ErrorCheck(cudaSetDevice(iDev), __FILE__, __LINE__);

// 由于大部分核函数返回值为空，所以采用内置函数检测核函数错误
// 在调用核函数后加入如下语句

// ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);
// ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);


void set_GPU(){
    int GPU_count = 0;
    cudaError_t error = cudaGetDeviceCount(&GPU_count);
    if(erroe != cudaSuccess){
        printf("No CUDA compatable GPU found\n");
        exit(-1);
    }
    else{
        printf("The count of GPUs is %d\n", GPU_count);
    }
    int Dev = 0;
    error = cudaSetDevice(Dev);
    if(error != cudaSuccess){
        printf("Fail to set GPU %d for computing\n", Dev);
        exit(-1);
    }
    else{
        printf("Set GPU %d for computing\n", Dev);
    }
}

// an example of cuda time evaluation

// int main(){
//     cudaEvent_t start, stop;
//     ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
//     ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
//     ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);
//     cudaEventQuery(start);

//     reducesum_GPU<<<grid, block>>>(g_A, g_B, n);

//     ErrorCheck(cudaEventRecord(stop), __FILE__, __LINE__);
//     ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);
//     float elapsed_time;
//     ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);
//     printf("Time = %g ms.\n", elapsed_time);
// }