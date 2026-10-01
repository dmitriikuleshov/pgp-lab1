#include "helper_cuda.h"
#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(double *arr1, double *arr2, double *result_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {
        result_arr[idx] = arr1[idx] > arr2[idx] ? arr1[idx] : arr2[idx];
        idx += offset;
    }
}

void cpu_max(double *arr1, double *arr2, double *result, int n) {
    for (int idx = 0; idx < n; ++idx) {
        result[idx] = arr1[idx] > arr2[idx] ? arr1[idx] : arr2[idx];
    }
}

double run_cpu_test(double *arr1, double *arr2, double *result, int n,
                    int repeats) {
    volatile double checksum = 0.0;
    auto start = std::chrono::steady_clock::now();

    for (int repeat = 0; repeat < repeats; ++repeat) {
        cpu_max(arr1, arr2, result, n);
        checksum += result[repeat % n];
    }

    auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count() /
           repeats;
}

void run_gpu_test(double *dev_arr1, double *dev_arr2, double *dev_result, int n,
                  int blocks, int threads, int repeats, double cpu_time_ms) {
    cudaEvent_t gpu_start, gpu_finish;
    checkCudaErrors(cudaEventCreate(&gpu_start));
    checkCudaErrors(cudaEventCreate(&gpu_finish));

    kernel<<<blocks, threads>>>(dev_arr1, dev_arr2, dev_result, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventRecord(gpu_start));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        kernel<<<blocks, threads>>>(dev_arr1, dev_arr2, dev_result, n);
        checkCudaErrors(cudaGetLastError());
    }
    checkCudaErrors(cudaEventRecord(gpu_finish));
    checkCudaErrors(cudaEventSynchronize(gpu_finish));

    float total_gpu_time_ms;
    checkCudaErrors(
        cudaEventElapsedTime(&total_gpu_time_ms, gpu_start, gpu_finish));
    float average_gpu_time_ms = total_gpu_time_ms / repeats;

    printf("n=%d blocks=%d threads=%d GPU=%.6f ms CPU=%.6f ms\n", n, blocks,
           threads, average_gpu_time_ms, cpu_time_ms);

    checkCudaErrors(cudaEventDestroy(gpu_start));
    checkCudaErrors(cudaEventDestroy(gpu_finish));
}

void run_test(int n) {
    int repeats = 100;
    size_t bytes = n * sizeof(double);

    double *arr1 = (double *)malloc(bytes);
    double *arr2 = (double *)malloc(bytes);
    double *result = (double *)malloc(bytes);

    for (int i = 0; i < n; ++i) {
        arr1[i] = (double)(rand() % 10000) / 10.0;
        arr2[i] = (double)(rand() % 10000) / 10.0;
    }

    double cpu_time_ms = run_cpu_test(arr1, arr2, result, n, repeats);

    double *dev_arr1, *dev_arr2, *dev_result;
    checkCudaErrors(cudaMalloc(&dev_arr1, bytes));
    checkCudaErrors(cudaMalloc(&dev_arr2, bytes));
    checkCudaErrors(cudaMalloc(&dev_result, bytes));
    checkCudaErrors(cudaMemcpy(dev_arr1, arr1, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_arr2, arr2, bytes, cudaMemcpyHostToDevice));

    run_gpu_test(dev_arr1, dev_arr2, dev_result, n, 1, 32, repeats,
                 cpu_time_ms);
    run_gpu_test(dev_arr1, dev_arr2, dev_result, n, 32, 32, repeats,
                 cpu_time_ms);
    run_gpu_test(dev_arr1, dev_arr2, dev_result, n, 64, 128, repeats,
                 cpu_time_ms);
    run_gpu_test(dev_arr1, dev_arr2, dev_result, n, 128, 256, repeats,
                 cpu_time_ms);
    run_gpu_test(dev_arr1, dev_arr2, dev_result, n, 256, 512, repeats,
                 cpu_time_ms);
    run_gpu_test(dev_arr1, dev_arr2, dev_result, n, 1024, 1024, repeats,
                 cpu_time_ms);

    checkCudaErrors(cudaFree(dev_arr1));
    checkCudaErrors(cudaFree(dev_arr2));
    checkCudaErrors(cudaFree(dev_result));
    free(arr1);
    free(arr2);
    free(result);
}

int main() {
    srand(42);

    run_test(1000);
    run_test(1000000);
    run_test(100000000);

    return 0;
}
