#include "plate_utils.h"
#include "stdio.h"
#include "cuda.h"
#include "cuda_runtime.h"

__global__ void cuda_hello() {
    printf("%d\n", blockIdx.x+1 + threadIdx.x);
    printf("hello from GPU  =======!\n");
}


void hello() {
    cuda_hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return;
}

__global__
void doSomething_kernel(unsigned char * gray, unsigned char * enhanced, unsigned char * smooth,
                        int dist, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > N) return;
    // pos = y*width + x
    int a, b, val, smt;
    val = gray[index];
    smt = smooth[index];
    if ((val - smt) > dist) smt = smt + (val - smt) * 0.5;
    smt = smt < 0.5 * dist ? 0.5 * dist : smt;
    b = smt + 0.5 * dist;
    b = b > 255 ? 255 : b;
    a = b - dist;
    a = a < 0 ? 0 : a;
    if(val >= a && val <= b )
        enhanced[index] = (int)(((val -a) / (0.5* dist)) * 255);
    else if (val < a)
        enhanced[index] = 0;
    else if (val > b)
        enhanced[index] = 255;
}


void doSomethingWith(unsigned char * gray, unsigned char * enhanced, unsigned char * smooth, int width, int height, int dist)
{
    // loop for all pixels
    // index should be from 0 to width*height - 1

    int N = width * height - 1;

    int blockSIze = 512;
    int numBlocks = (N + blockSIze -1 )/ blockSIze;
    doSomething_kernel<<<numBlocks, blockSIze>>>(gray, enhanced, smooth, dist, N);

    cudaDeviceSynchronize();
    return;
}
