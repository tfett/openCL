#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "matrix_utility.c"

#define BLOCK_SIZE 16
#define DEBUG_PRINT 0


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
    double fillMatrixTime, cpuAllocateTime, cpuExecuteTime, totalHostTime, totalTime;
    
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
    
    // multiply on the host
    double host_start = getTime();
    cpuMatrixMultiply(A,B,check,Ah,Aw,Bw,&cpuExecuteTime);
    double host_stop = getTime();
    totalHostTime = host_stop - host_start;

    // display the matrices
    if(DEBUG_PRINT) {
        id = "A";
        printMatrix(Ah,Aw,id,A);
        id = "B";
        printMatrix(Bh,Bw,id,B);
    }


    free(A);
    free(B);
    free(results);

    double total_stop   = getTime();
    totalTime           = total_stop - total_start;

    if(1) {
        printf("%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\n",Ah,Aw,Bh,Bw,Ah,Bw, fillMatrixTime, cpuAllocateTime, cpuExecuteTime, totalTime);
        
    } else {
        printf("The matrices do not match!\n");
    }

    return 0;

}