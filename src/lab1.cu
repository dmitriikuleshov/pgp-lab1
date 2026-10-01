#include "helper_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(double *arr1, double *arr2, double *result_arr, int n) {
    /*
        threadIdx.x
        blockIdx.x
        blockDim.x
        gridDim.x
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    // printf("threadIdx.x=%d, blockIdx.x=%d, idx=%d\n", threadIdx.x,
    // blockIdx.x, idx);
    while (idx < n) {
        result_arr[idx] = arr1[idx] > arr2[idx] ? arr1[idx] : arr2[idx];
        idx += offset;
    }
}

int main() {
    int i, n;

    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "Failed to read n\n");
        return 1;
    }

    double *arr1 = (double *)malloc(sizeof(double) * n);
    double *arr2 = (double *)malloc(sizeof(double) * n);
    double *result_arr = (double *)malloc(sizeof(double) * n);

    for (int i = 0; i < n; ++i) {
        if (scanf("%lf", &arr1[i]) != 1) {
            fprintf(stderr, "Failed to read arr1[%d]\n", i);
            return 1;
        }
    }

    for (int i = 0; i < n; ++i) {
        if (scanf("%lf", &arr2[i]) != 1) {
            fprintf(stderr, "Failed to read arr2[%d]\n", i);
            return 1;
        }
    }

    double *dev_arr1, *dev_arr2, *dev_result_arr;

    // Allocates gpu memory
    checkCudaErrors(cudaMalloc(&dev_arr1, sizeof(double) * n));
    checkCudaErrors(cudaMalloc(&dev_arr2, sizeof(double) * n));
    checkCudaErrors(cudaMalloc(&dev_result_arr, sizeof(double) * n));

    checkCudaErrors(
        cudaMemcpy(dev_arr1, arr1, sizeof(double) * n, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(dev_arr2, arr2, sizeof(double) * n, cudaMemcpyHostToDevice));

    // Function that is being processed on gpu
    kernel<<<10, 10>>>(dev_arr1, dev_arr2, dev_result_arr, n);

    // Check errors in kernel
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(result_arr, dev_result_arr, sizeof(double) * n,
                               cudaMemcpyDeviceToHost));

    for (i = 0; i < n; i++)
        printf("%f ", result_arr[i]);
    printf("\n");

    checkCudaErrors(cudaFree(dev_arr1));
    checkCudaErrors(cudaFree(dev_arr2));
    checkCudaErrors(cudaFree(dev_result_arr));
    free(arr1);
    free(arr2);
    free(result_arr);
    return 0;
}
