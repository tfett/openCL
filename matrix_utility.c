#include <sys/time.h>
#include <stdio.h>
#include <math.h>

#define MAX_RAND 25

// Print a matrix...if it's not too big
void printMatrix(int h,int w, char *id, float *matrix){
    if(h <= 20 && w <=20) {
        printf("\nMatrix %s: %dx%d\n",id,h,w);
            for(int m=0;m<h;m++){
                for(int n=0;n<w;n++){
                    printf("%7.2f ",matrix[m*w+n]);
                }
                printf("\n");
            }
    } else {
        printf("Matrix %s: %dx%d is too large to display.\n",id,h,w);
    }
}

// Fill a matrix with random values
void fillMatrix(int h, int w, float *matrix) {
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            matrix[i * w + j] = (float)rand()/RAND_MAX;
        }
    }
}

// Check if two matrices are the same - no output if same
int compareMatrices(const float *A, const float *B, int Width, int Height) {

   int flag = 1;
   for(int i = 0; i < Height; i++) {
      for(int j = 0; j < Width; j++) {
        int idx = i * Width + j;
        if(fabs(A[idx] - B[idx]) > 0.01f) {
            flag = 0;
            break;
         }
      }
   }
    return flag;
}

// Returns seconds and microseconds since Epoch as a double
double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec+(tv.tv_usec/1000000.0);
}

// multiply two matrices on the cpu
void cpuMatrixMultiply(const float *mA, const float *mB, float *results, int Ah, int Aw, int Bw, double *cpuExecuteTime) {
    
    double multiply_start = getTime();
    
    for (int i = 0; i < Ah; ++i)
        for (int j = 0; j < Bw; ++j) {
            float total = 0;
            for (int k = 0; k < Aw; ++k) {
                float vA = mA[i * Aw + k];
                float vB = mB[k * Bw + j];
                total += vA * vB;
            }
            results[i * Bw + j] = total;
        }
        
    double multiply_stop    = getTime();
    *cpuExecuteTime          = multiply_stop - multiply_start;
}

// Round Up Division function
size_t roundUp(int group_size, int global_size) 
{
    int r = global_size % group_size;
    if(r == 0) 
    {
        return global_size;
    } else 
    {
        return global_size + group_size - r;
    }
}