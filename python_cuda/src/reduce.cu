#include<torch/extension.h>
#include<vector>
#include<cuda.h>
#include<cuda_runtime.h>

template <typename T>
__global__ void reduce_sum(T *input, T *output, int n){
    extern __shared__ T mm[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    mm[tid] = (id < n) ? input[id] : 0;
    __syncthread;

    for(int i = blockDim.x >> 1 ; s >= 1 ; i >> 1){
        if(tid < i){
            mm[tid] += mm[tid+i];
        }
        __syncthread;
    }
    if(tid == 0){
        output[blockIdx.x] = mm[0];
    }
}

torch::Tensor reduce_cuda(torch::Tensor input){
    const int num_elements = input.size(0);
    const int threads = 1024;
    const int blocks = (num_elements + threads - 1) / threads;

    auto option = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({blocks}, options);

    reduce_sum<<<blocks, threads, blocks*sizeof(float)>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_elements);

    while(block > 1){
        int new_num_elements = blocks;
        int new_blocks = (new_num_elements + threads - 1) / threads;
        torch::Tensor new_output = torch::empty({new_blocks}, options);

        reduce_sum<<<new_blocks, threads, new_blocks*sizeof(float)>>>(
            output.data_ptr<float>(), 
            new_output.data_ptr<float>(), 
            new_num_elements);
        
        output = new_output;
        blocks = new_blocks;
    }
    return output[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("reduce", &reduce_cuda, "Reduce operation CUDA");
}