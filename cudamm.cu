#include <assert.h>
#include "matrix_utility.c"

#define TILE_WIDTH 16
#define BLOCK_SIZE 16
#define DEBUG_PRINT 0

/* Compile with nvcc -gencode=arch=compute_30,code=sm_30 cudamm.cu -o cudamm */

__global__ void MM_kernel(const float *mA, const float *mB, float *results, int Ah, int Aw, int Bw){

    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float result = 0.0f;

    for (int k = 0; k < ceil(Aw/(double)TILE_WIDTH); k++) {
        if (k * TILE_WIDTH + tx < Aw && Row < Ah) {
            Ads[ty][tx] = mA[Row * Aw + k * TILE_WIDTH + tx];
        } else {
            Ads[ty][tx] = 0.0f;
        }

        if (k * TILE_WIDTH + ty < Aw && Col < Bw) {
            Bds[ty][tx] = mB[(k * TILE_WIDTH + ty) * Bw + Col];
        } else {
            Bds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int n = 0; n < TILE_WIDTH; ++n) { 
            result += Ads[ty][n] * Bds[n][tx];
        }
        __syncthreads();
    }

    if (Row < Ah && Col < Bw) {
        results[((by * blockDim.y + ty)*Bw)+(bx*blockDim.x)+tx] = result;
    }
}

void MM_dev(const float *mA, const float *mB, float *results, int Ah, int Aw, int Bw, double *gpuAllocTime, double *gpuCopyTime, double *gpuExecuteTime){
    //Allocate memory
    double alloc_start = getTime(); // Performance timer
    float *mA_dev,*mB_dev,*results_dev;
    
    assert(cudaMalloc((void**) &mA_dev,sizeof(float)*(Ah*Aw)) == cudaSuccess);
    assert(cudaMalloc((void**) &mB_dev,sizeof(float)*(Aw*Bw)) == cudaSuccess);
    assert(cudaMalloc((void**) &results_dev,sizeof(float)*(Ah*Bw)) == cudaSuccess);
    
    double alloc_stop       = getTime(); // Performance timer
    double copy_in_start    = getTime(); // Performance timer
    
    //copy the input matrices to the device
    assert(cudaMemcpy(mA_dev,mA,sizeof(float)*(Ah*Aw),cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(mB_dev,mB,sizeof(float)*(Aw*Bw),cudaMemcpyHostToDevice) == cudaSuccess);
    
    double copy_in_stop = getTime(); // Performance timer
    double kernel_start = getTime(); // Performance timer
    
    //invoke the kernel
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 dimGrid(ceil(Bw/(double)dimBlock.x),ceil(Ah/(double)dimBlock.y),1);
    MM_kernel<<<dimGrid,dimBlock>>>(mA_dev,mB_dev,results_dev,Ah,Aw,Bw);

    double kernel_stop      = getTime(); // Performance timer
    double copy_out_start   = getTime(); // Performance timer
    
    //retrieve results
    assert(cudaMemcpy(results,results_dev,sizeof(float)*(Ah*Bw),cudaMemcpyDeviceToHost) == cudaSuccess);

    double copy_out_stop = getTime(); // Performance timer
    
    *gpuAllocTime    = alloc_stop - alloc_start;
    *gpuCopyTime     = copy_in_stop - copy_in_start + copy_out_stop - copy_out_start;
    *gpuExecuteTime  = kernel_stop - kernel_start;
    
    //free device memory
    cudaFree(mA_dev);
    cudaFree(mB_dev);
    cudaFree(results_dev);    
}

int main(int argc, char *argv[]){

    if(argc < 5){
        printf("You forgot to enter the matrix dimensions!\n");
        return 0;
    }
    if(argc > 5){
        printf("Too many arguments!\n");
        return 0;
    }
    
    srand(time(NULL));
    // Timer result variables
    double gpuAllocateTime, gpuCopyTime, gpuExecuteTime, fillMatrixTime, cpuAllocateTime, totalDeviceTime, totalTime;
    
    double total_start = getTime();
    
    // store command line arguments as the matrix dimensions
    int Ah = atoi(argv[1]);
    int Aw = atoi(argv[2]);
    int Bh = atoi(argv[3]);
    int Bw = atoi(argv[4]);
    if(Ah % BLOCK_SIZE || Aw % BLOCK_SIZE || Bh % BLOCK_SIZE || Bw % BLOCK_SIZE) {
        printf("Inputs must be multiples of %d.\n", BLOCK_SIZE);
        return 0;
    }
    
    //if the given dimensions are not multiply-able give error and exit
    if(Aw != Bh){
        printf("Cannot compute the matrix multiplication given the matrix dimensions\n");
        return 0;
    }
    
    double cpu_allocate_start = getTime();
    
    // setup matrices
    float *A, *B, *results, *check;
    char *id;    

    A = (float*)malloc(sizeof(float)*Ah*Aw); assert(A != 0);
    B = (float*)malloc(sizeof(float)*Bh*Bw); assert(B != 0);
    results = (float*)malloc(sizeof(float)*Ah*Bw); assert(results != 0);
    check = (float*)malloc(sizeof(float)*Ah*Bw); assert(check != 0);
    
    double cpu_allocate_stop    = getTime();
    cpuAllocateTime             = cpu_allocate_stop - cpu_allocate_start;
    double fill_start           = getTime();
    
    // fill the matrices with random values
    fillMatrix(Ah, Aw, A);
    fillMatrix(Bh, Bw, B);
    
    double fill_stop = getTime();
    fillMatrixTime   = fill_stop - fill_start;
    
    // display the matrices
    if(DEBUG_PRINT) {
        id = "A";
        printMatrix(Ah,Aw,id,A);
        id = "B";
        printMatrix(Bh,Bw,id,B);
    }
    
    // multiply on the device
    double device_start = getTime();
    MM_dev(A,B,results,Ah,Aw,Bw,&gpuAllocateTime,&gpuCopyTime,&gpuExecuteTime);
    double device_stop = getTime();
    totalDeviceTime = device_stop - device_start;
    
    // multiply on the host
    //double host_start = getTime();
    //cpuMatrixMultiply(A,B,check,Ah,Aw,Bw,&cpuExecuteTime);
    //double host_stop = getTime();
    //totalHostTime = host_stop - host_start;
    
    // display the results
    if(DEBUG_PRINT) {
        id = "results";
        printMatrix(Ah,Bw,id,results);
        id = "check";
        printMatrix(Ah,Bw,id,check);
    }
    
    // compare the matrices and display a message if they are not identical
    //double compare_start = getTime();
    ///int isSame = compareMatrices(results, check, Bw, Ah);
    //double compare_stop = getTime();
    //compareTime = compare_stop - compare_start;
    
    free(A);
    free(B);
    free(results);
    free(check);
    
    double total_stop   = getTime();
    totalTime           = total_stop - total_start;
    
    if(1) {
        printf("%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",Ah,Aw,Bh,Bw,Ah,Bw,gpuAllocateTime, gpuCopyTime, gpuExecuteTime, fillMatrixTime, cpuAllocateTime, totalDeviceTime, totalTime);
    } else {
        printf("The matrices do not match!\n");
    }
    
    return 0;
}
