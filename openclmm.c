#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "matrix_utility.c"

#define PROGRAM_FILE "openclmm.cl"
#define MM_KERNEL "MM_kernel"
#define BLOCK_SIZE 16
#define DEBUG_PRINT 0

void MM_dev(float *mA, float *mB, float *results, int Ah, int Aw, int Bw, double *gpuAllocTime, double *gpuCopyTime, double *gpuExecuteTime){
    
    double setup_start = getTime(); // Performance timer
    // Error code variable
    cl_int errcode_ret;

    
    // Get the platform
    cl_platform_id platform_id = NULL;
    if((clGetPlatformIDs(1, &platform_id, NULL)) != CL_SUCCESS) {
        perror("Unable to get the platform ID\n");
        exit(1);
    }
    
    // Get the devices
    cl_device_id device_id = NULL;
    if((clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL)) != CL_SUCCESS) {
        perror("Unable to get the device ID\n");
        exit(1);
    }
    
    // Create the context
    cl_context context;
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &errcode_ret);
    if(errcode_ret != CL_SUCCESS) {
        perror("Unable to create a context\n");
        exit(1);
    }
    
    // Create the command queue
    cl_command_queue command_queue = NULL;
    command_queue = clCreateCommandQueue(context, device_id, 0, &errcode_ret);
    if(errcode_ret != CL_SUCCESS) {
        perror("Unable to create a command queue\n");
        exit(1);
    }
    
    // Load the kernel program file
    FILE *program_file;
    char *program_source;
    size_t program_size;
    program_file = fopen(PROGRAM_FILE, "r");
    if(program_file == NULL) {
        perror("Unable to load kernel file\n");
        exit(1);
    }
    
    
    
    fseek(program_file, 0, SEEK_END);
    program_size = ftell(program_file);
    rewind(program_file);
    program_source = (char*)malloc(program_size + 1);
    program_source[program_size] = '\0';
    fread(program_source, sizeof(char), program_size, program_file);
    fclose(program_file);
    
    // Create the program
    cl_program program;
    program = clCreateProgramWithSource(context, 1,(const char**)&program_source, &program_size, &errcode_ret);
    if(errcode_ret != CL_SUCCESS) {
        perror("Unable to create the program\n");
        exit(1);
    }
    free(program_source);
    
    // Build program
    errcode_ret = (clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL));
    if(errcode_ret != CL_SUCCESS) {
        printf("OpenCL Error # %d\n",errcode_ret);
        printf("Unable to build the program\n");
        exit(1);
    }
    
    double setup_stop = getTime(); // Performance timer
    double alloc_start = getTime(); // Performance timer
    
    // Allocate device buffers
    cl_mem mA_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*(Ah*Aw), NULL, &errcode_ret);
    if(errcode_ret != CL_SUCCESS) {
        perror("Unable to create buffer mA_dev\n");
        exit(1);
    }
    cl_mem mB_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*(Aw*Bw), NULL, &errcode_ret);
    if(errcode_ret != CL_SUCCESS) {
        perror("Unable to create buffer mB_dev\n");
        exit(1);
    }
    cl_mem results_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*(Ah*Bw), NULL, &errcode_ret);
    if(errcode_ret != CL_SUCCESS) {
        perror("Unable to create buffer results_dev\n");
        exit(1);
    }
    
    double alloc_stop       = getTime(); // Performance timer
    double copy_in_start    = getTime(); // Performance timer
    
    // Copy to buffers
    if(clEnqueueWriteBuffer(command_queue, mA_dev, CL_TRUE, 0, sizeof(float)*(Ah*Aw), (const void *)mA, 0, 0, NULL) != CL_SUCCESS) {
        perror("Unable to copy to buffer mA_dev\n");
        exit(1);
    }
    if(clEnqueueWriteBuffer(command_queue, mB_dev, CL_TRUE, 0, sizeof(float)*(Aw*Bw), (const void *)mB, 0, 0, NULL) != CL_SUCCESS) {
        perror("Unable to copy to buffer mB_dev\n");
        exit(1);
    }
    
    double copy_in_stop = getTime(); // Performance timer
    double kernel_start = getTime(); // Performance timer
    
    // Create kernel
    cl_kernel kernel = NULL;
    kernel = clCreateKernel(program, MM_KERNEL, &errcode_ret);
    if(errcode_ret != CL_SUCCESS) {
        perror("Unable to create kernel\n");
        exit(1);
    }
    
    
    // Set kernel arguments
    errcode_ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&results_dev);
    errcode_ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mA_dev);
    errcode_ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mB_dev);
    errcode_ret |= clSetKernelArg(kernel, 3, sizeof(float) * BLOCK_SIZE *BLOCK_SIZE, 0);
    errcode_ret |= clSetKernelArg(kernel, 4, sizeof(float) * BLOCK_SIZE *BLOCK_SIZE, 0);
    errcode_ret |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&Aw);
    errcode_ret |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&Bw);
    errcode_ret |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void*)&Ah);
    if(errcode_ret != CL_SUCCESS) {
        perror("Unable to set kernel arguments.\n");
        exit(1);
    }
    
    // Set up grid and blocks
    size_t dimBlock[2], dimGrid[2];
    dimBlock[0] = BLOCK_SIZE;
    dimBlock[1] = BLOCK_SIZE;
    dimGrid[0] = roundUp(BLOCK_SIZE, Bw);
    dimGrid[1] = roundUp(BLOCK_SIZE, Ah);
    
    
    //Launch the kernel
    errcode_ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, dimGrid, dimBlock, 0, NULL, NULL);
    if(errcode_ret != CL_SUCCESS) {
        printf("OpenCL Error # %d\n",errcode_ret);
        printf("Kernel failed to execute\n");
        exit(1);
    }
    
    clFlush(command_queue);
    clFinish(command_queue);
    
    double kernel_stop      = getTime(); // Performance timer
    double copy_out_start   = getTime(); // Performance timer
    
    // Get results
    if(clEnqueueReadBuffer(command_queue, results_dev, CL_FALSE, 0, sizeof(float)*(Ah*Bw), results, 0, NULL, NULL) != CL_SUCCESS) {
        perror("Unable to copy from buffer results_dev.\n");
        exit(1);
    }
    
    double copy_out_stop = getTime(); // Performance timer
    
    *gpuAllocTime    = setup_stop - setup_start + alloc_stop - alloc_start;
    *gpuCopyTime     = copy_in_stop - copy_in_start + copy_out_stop - copy_out_start;
    *gpuExecuteTime  = kernel_stop - kernel_start;
    
    // Free resources
    
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(mA_dev);
    clReleaseMemObject(mB_dev);
    clReleaseMemObject(results_dev);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    


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
    
    //compare the matrices and display a message if they are not identical
    //double compare_start = getTime();
    //int isSame = compareMatrices(results, check, Bw, Ah);
    //double compare_stop = getTime();
    //compareTime = compare_stop - compare_start;
    
    free(A);
    free(B);
    free(results);
    free(check);
    
    double total_stop   = getTime();
    totalTime           = total_stop - total_start;
    
    //removed comparison checking for running tests
    if(1) {
        printf("%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",Ah,Aw,Bh,Bw,Ah,Bw,gpuAllocateTime, gpuCopyTime, gpuExecuteTime, fillMatrixTime, cpuAllocateTime, totalDeviceTime,  totalTime);
        
    } else {
        printf("The matrices do not match!\n");
    }
    
    return 0;
}