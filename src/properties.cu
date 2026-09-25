#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    printf("Graphics Processor: %s\n", p.name);
    printf("Compute capability: %d.%d\n", p.major, p.minor);
    printf("Graphics Memory: %.0f MiB; available CUDA %zu bytes\n",
           p.totalGlobalMem / 1048576.0, freeMem);
    printf("Shared memory per block: %zu bytes (%zu KiB)\n",
           p.sharedMemPerBlock, p.sharedMemPerBlock / 1024);
    printf("Constant memory: %zu bytes (%zu KiB)\n", p.totalConstMem,
           p.totalConstMem / 1024);
    printf("Registers per block: %d\n", p.regsPerBlock);
    printf("Maximum number of threads per block: %d\n", p.maxThreadsPerBlock);
    printf("Maximum grid size: %d x %d x %d blocks\n", p.maxGridSize[0],
           p.maxGridSize[1], p.maxGridSize[2]);
    printf("Number of multiprocessors: %d\n", p.multiProcessorCount);

    return 0;
}