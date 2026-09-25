#include <chrono>
#include <cuda_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void kernel(float *arr1, float *arr2, float *result_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {
        result_arr[idx] = arr1[idx] > arr2[idx] ? arr1[idx] : arr2[idx];
        idx += offset;
    }
}

void cpu_max(float *arr1, float *arr2, float *result, int n) {
    for (int idx = 0; idx < n; ++idx) {
        result[idx] = arr1[idx] > arr2[idx] ? arr1[idx] : arr2[idx];
    }
}

int main() {

    int array_sizes[3] = {1000, 1000000, 100000000};

    int block_counts[6] = {1, 32, 64, 128, 256, 1024};
    int thread_counts[6] = {32, 32, 128, 256, 512, 1024};

    int repeats = 100;

    for (int n : array_sizes) {

        size_t bytes = n * sizeof(float);

        float *arr1 = (float *)malloc(bytes);
        float *arr2 = (float *)malloc(bytes);
        float *result = (float *)malloc(bytes);

        std::random_device rd;
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 1000.0f);

        for (int i = 0; i < n; ++i) {
            arr1[i] = dist(gen);
            arr2[i] = dist(gen);
        }

        // Measure average CPU time
        volatile float checksum = 0.0f;
        auto start = std::chrono::steady_clock::now();

        for (int repeat = 0; repeat < repeats; ++repeat) {
            cpu_max(arr1, arr2, result, n);
            // Use the result (so the compiler won't skip it)
            checksum += result[repeat % n];
        }

        auto finish = std::chrono::steady_clock::now();

        double cpu_time_ms =
            std::chrono::duration<double, std::milli>(finish - start).count() /
            repeats;

        float *device_arr1;
        float *device_arr2;
        float *device_result;

        cudaMalloc(&device_arr1, bytes);
        cudaMalloc(&device_arr2, bytes);
        cudaMalloc(&device_result, bytes);

        cudaMemcpy(device_arr1, arr1, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(device_arr2, arr2, bytes, cudaMemcpyHostToDevice);

        cudaEvent_t gpu_start;
        cudaEvent_t gpu_finish;
        cudaEventCreate(&gpu_start);
        cudaEventCreate(&gpu_finish);

        for (int config = 0; config < 6; ++config) {
            int blocks = block_counts[config];
            int threads = thread_counts[config];

            // GPU warm-up
            kernel<<<blocks, threads>>>(device_arr1, device_arr2, device_result,
                                        n);
            cudaDeviceSynchronize();

            cudaEventRecord(gpu_start);

            for (int repeat = 0; repeat < repeats; repeat++) {
                kernel<<<blocks, threads>>>(device_arr1, device_arr2,
                                            device_result, n);
            }

            cudaEventRecord(gpu_finish);
            cudaEventSynchronize(gpu_finish);

            float total_gpu_time_ms;
            cudaEventElapsedTime(&total_gpu_time_ms, gpu_start, gpu_finish);

            float average_gpu_time_ms = total_gpu_time_ms / repeats;

            printf("n=%d blocks=%d threads=%d GPU=%.6f ms CPU=%.6f ms\n", n,
                   blocks, threads, average_gpu_time_ms, cpu_time_ms);
        }

        cudaEventDestroy(gpu_start);
        cudaEventDestroy(gpu_finish);

        cudaFree(device_arr1);
        cudaFree(device_arr2);
        cudaFree(device_result);

        free(arr1);
        free(arr2);
        free(result);
    }

    return 0;
}
