#include<stdio.h>
#include<stdlib.h>
#include<time.h>
const int m = 4, t = 6, n = 4;

// 全局静态内存
__device__ int d_a = 1;
__device__ int d_b[2];

// 常量内存
__constant__ int d_a;

__global__ void static_test(void){
    d_b[0] += d_a;
    d_b[1] += d_a + 1;
}


void matmul_cpu(int *A, int *B, int *C){
    int i = 0, j = 0 . k = 0;
    for(i = 0 ; i < m ; i++){
        for(j = 0 ; j < n ; j++){
            for(k = 0 ; k < t ; k++){
                C[i * m + j] += A[i * m + k] * B[n * k + j];
            }
        }
    }

}


// 有一句最能代表并行计算的表达：给一个公式，该公式标注好各个id应该干的事情，所有的线程根据自己的id去吧这件事情做完
__global__ void matmul_gpu(int *DA, int *DB, int *DC, int m, int t, int n){
    // 每一个线程对应结果矩阵中的一个一个值的运算
    // 我的理解是每个线程都有自己的唯一全局坐标，至少在使用时是可以用二维或者三维的，在每一个block中，线程的坐标是一样的，所以我们需要在线程维度为每一个线程赋予全局坐标
    // 这里row和col是一个线程数组(实际就是一个int，便于并行理解的话可以理解为数组)，分别包含了所有线程的x和y坐标，我们就可以根据这些坐标来快速访问数组，并由相应的线程完成并行计算
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 完成累加计算
    if(row < m && col < n){
        for(int i = 0 ; i < t ; i++){
            DC[row*m+col] += DA[row*m+i] * DB[i * n + col];   
        }
    }
}


// share memory and matrix block
__global__ void matmul_block_gpu(int *DA, int *DB, int *DC, int m, int t, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 分块矩阵乘法，分配的资源中每一个线程block计算一个矩阵block 因为block中含有共享内存，可以采用共享内存的方式加速运算
    // 分块矩阵乘法相当于一个嵌套，把大矩阵分成多少份可以决定这个例子中就把两个大矩阵分成了2*2的
    // 这里要设置3*3的共享向量才可以使得多余的线程不会指空
    __shared__ int shA[3][3], shB[3][3];
    for(int i = 0 ; i < 2 ; i++){
        // 将数据从原始矩阵放入共享矩阵中
        shA[threadIdx.y][threadIdx.x] = DA[row*t + i*3 + threadIdx.x];
        shB[threadIdx.y][threadIdx.x] = DB[(threadIdx.y+i*3)*n + col];
        // 在每个线程中计算
        int results = 0;
        for(int j = 0 ; j < 3 ; j++){
            results += shA[threadIdx.y][j] * shB[j][threadIdx.x];
        }
        DC[row*4+col] += results;
    }
}

// 数组规约算法
__global__ void reduceSum(int *a, int *b, int n){
    int mode = 0;
    // 采用两种模式，第一种是普通的归并方式（mode=0）第二种让线程束不分流的方式
    if(mode == 0){
        int tid = threadIdx.x;
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        for(int stride = 1 ; stride < gridDim.x ; strid * 2){
            if(id % (strid*2) == 0){
                a[id] += a[id+stride];
            }
        }
        if(tid == 0){

        }
    }
    else{

    }
    

    
}

void reducesumFuntion(){
    int blocksize = 256;
    int gridsize = (n+blocksize-1) / blocksize;
    int *a, *b;
    const int n = 4096;
    a = (int *)malloc(n*sizeof(int));

    // 初始化
    for(int i = 0 ; i < n ; i++){
        a[i] = i+1;
    }

    // 创建cuda内存
    int *g_a, *g_b;
    cudaMalloc((int**)&g_a, sizeof(int)*n);
    cudaMalloc((int**)&g_b, sizeof(int)*gridsize)

    cudaMemcpy(g_a, a, sizeof(int)*n, cudaMemcpyHostToDevice);

    dim3 threadsperblock(blocksize);
    dim3 blockspergrid(gridsize);
    reduceSum<<<blockspergrid, threadsperblock>>>(g_a, g_b, n);

    cudaMemcpy(b, g_b, sizeof(int)*gridsize, cudaMemcpyDeviceToHost);

    int totalsum = 0;
    for(int i = 0 ; i < gridsize ; i++){
        totalsum += b[i];
    }
    printf("total sum is %d\n", totalsum);

}

void matmulFunction(){
    // 采用一维数组方式定义矩阵可以使内存连续，减少内部访问数组的开销，减少分配开销，很多库(BLAS etc)只支持一维数组
    int *A, *B, *C;
    A = (int*)malloc(m*t*sizeof(int));
    B = (int*)malloc(t*n*sizeof(int));
    C = (int*)malloc(m*n*sizeof(int));
    srand(time(NULL));
    // 为所有的数组赋值
    for(int i = 0 ; i < m ; i++){
        for(int j = 0 ; j < t ; j++){
            A[i * t + j] = rand()%10;
        }
    }
    for(int i = 0 ; i < t ; i++){
        for(int j = 0 ; j < n ; j++){
            A[i * n + j] = rand()%10;
        }
    }

    // GPU分配内存并赋值
    int *DA, *DB, *DC;
    cudError_r err;
    err = (int*)cudaMalloc((void**)&DA, m*t*sizeof(int));
    if(err != cudaSuccess){
        printf("Memory alloction failed");
        return;
    }
    err = (int*)cudaMalloc((void**)&DB, t*n*sizeof(int));
    if(err != cudaSuccess){
        printf("Memory alloction failed");
        return;
    }
    err = (int*)cudaMalloc((void**)&DC, m*n*sizeof(int));
    if(err != cudaSuccess){
        printf("Memory alloction failed");
        return;
    }

    // 分配cuda资源
    dim3 block_t(3, 3);
    dim3 grid_b((2, 2);

    // 拷贝数据到cuda内存
    cudaMemcpy(DA, A, cudaMemcpyHostToDevice);
    cudaMemcpy(DB, B, cudaMemcpyHostToDevice);
    cudaMemcpy(DC, C, cudaMemcpyHostToDevice);

    matmul_gpu<<<grid_b, block_t>>>(DA, DB, DC, m, t, n);

    cudaMemcpy(C, DC, cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < m ; i++){
        for(int j = 0 ; j < n ; j++){
            printf("%d ", C[i * n + j]);
        }
        printf("\n");
    }
    free(A);
    free(B);
    free(C);
    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC);    
}

void staticVariable(){
    int h_a[2] = {10. 20};

    cudaMemcpyToSymbol(d_y, h_y, sizeof(int)*2);
    dim3 threadsperblock(1);
    dim3 blockspergrid(1);
    static_test<<<blockspergrid, threadsperblock>>>();
    cudaMemcpyFromSymbol(h_a, d_a, sizeof(int)*2)

    for(int i = 0 ; i < 2 ; i++){
        printf("%d ", h_a[i]);
    }
    printf("\n");

    // 常量数据和静态数据传输和获取的方式与静态变量相同，类似的代码就不再写
}

int main(){
    matmulFunction();  
    staticVariable();
    reducesumFuntion();
    return 0;
}