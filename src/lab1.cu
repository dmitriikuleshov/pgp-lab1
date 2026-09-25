#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(float *arr1, float *arr2, float *result_arr, int n) {
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

    float *arr1 = (float *)malloc(sizeof(float) * n);
    float *arr2 = (float *)malloc(sizeof(float) * n);
    float *result_arr = (float *)malloc(sizeof(float) * n);

    for (int i = 0; i < n; ++i) {
        if (scanf("%f", &arr1[i]) != 1) {
            fprintf(stderr, "Failed to read arr1[%d]\n", i);
            return 1;
        }
    }

    for (int i = 0; i < n; ++i) {
        if (scanf("%f", &arr2[i]) != 1) {
            fprintf(stderr, "Failed to read arr2[%d]\n", i);
            return 1;
        }
    }

    float *dev_arr1, *dev_arr2, *dev_result_arr;

    // Allocates gpu memory
    cudaMalloc(&dev_arr1, sizeof(float) * n);
    cudaMalloc(&dev_arr2, sizeof(float) * n);
    cudaMalloc(&dev_result_arr, sizeof(float) * n);

    cudaMemcpy(dev_arr1, arr1, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_arr2, arr2, sizeof(float) * n, cudaMemcpyHostToDevice);

    // Function that is being processed on gpu
    kernel<<<10, 10>>>(dev_arr1, dev_arr2, dev_result_arr, n);

    cudaMemcpy(result_arr, dev_result_arr, sizeof(float) * n,
               cudaMemcpyDeviceToHost);

    for (i = 0; i < n; i++)
        printf("%f ", result_arr[i]);
    printf("\n");

    cudaFree(dev_arr1);
    cudaFree(dev_arr2);
    cudaFree(dev_result_arr);
    free(arr1);
    free(arr2);
    free(result_arr);
    return 0;
}