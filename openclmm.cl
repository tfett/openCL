#define BLOCK_SIZE 16
#define TILE_WIDTH 16
#define ADS(i, j) Ads[j + i * BLOCK_SIZE]
#define BDS(i, j) Bds[j + i * BLOCK_SIZE]

__kernel void
MM_kernel( __global float* results, __global float* mA, __global float* mB, 
	   __local float* Ads, __local float* Bds, int Aw, int Bw, int Ah)
{

    int bx = get_group_id(0);
    int by = get_group_id(1);
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    
    int Row = get_global_id(1);
    int Col = get_global_id(0);
    
    int aBegin = Aw * TILE_WIDTH * by;
    int aEnd   = aBegin + Aw - 1;
    int aStep  = TILE_WIDTH;
    int bBegin = TILE_WIDTH * bx;
    int bStep  = TILE_WIDTH * Bw;
    float result = 0.0f;
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        ADS(ty, tx) = mA[a + Aw * ty + tx];
        BDS(ty, tx) = mB[b + Bw * ty + tx];

        barrier(CLK_LOCAL_MEM_FENCE);
       
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            result += ADS(ty, k) * BDS(k, tx);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (Row < Ah && Col < Bw) {
        results[Row * get_global_size(0) + Col] = result;
    }
}

