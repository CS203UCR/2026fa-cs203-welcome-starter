#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include "microbench_cuda.h"
#define BLOCK_SIZE 1024


extern "C" 
uint64_t *baseline_double_cuda(uint64_t * _array, unsigned long  size)
{
     double *d_array, *d_output;
     cudaMalloc((void **) &d_array, sizeof(double)*size);
     cudaMalloc((void **) &d_output, sizeof(double)*size);
     cudaMemcpy(d_array, _array, sizeof(double)*size, cudaMemcpyHostToDevice);

     unsigned int grid_cols = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;


     baseline_double_cuda_kernel<<<grid_cols, BLOCK_SIZE>>>(d_output, d_array, size);

     // Transefr results from device to host 
     cudaMemcpy(_array, d_output, sizeof(double)*size, cudaMemcpyDeviceToHost);
     cudaFree(d_output);
     cudaFree(d_array);
     return _array;
}
__global__ void baseline_double_cuda_kernel(double *output, double *vector, unsigned long  size) {
    // Calculate global thread index
    unsigned long int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure thread is within bounds of the array
    if (i < size) {
        output[i] = sqrt(vector[i]);
    }
}

__global__ void baseline_double_cuda_init(double *array, int size) 
{
    unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos < size) 
    {
        array[pos] = 0;
    }
}

extern "C"  uint64_t* calculate_distance_matrix(uint64_t *_matrix,unsigned long int size) {
    int rows = sqrt(size);
    int cols = rows;
    double *matrix = (double *)_matrix;
    double *distance_matrix = (double *)malloc(sizeof(double)*size);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            double sum_squares = 0.0;
            
            // Calculate the sum of squared differences between row i and row j
            for (int k = 0; k < cols; k++) {
                double diff = matrix[i*cols+k] - matrix[j*cols+k];
                sum_squares += diff * diff;
            }
            
            // Take the square root to get the Euclidean distance
            distance_matrix[i*cols+j] = sum_squares;
        }
    }
    return (uint64_t*)distance_matrix;
}
//START
// CUDA Kernel to calculate the Euclidean distance matrix
__global__ void distance_matrix_kernel(const double *matrix, double *distance_matrix, int rows, int cols) {
    // Map thread indices to output matrix row (i) and column (j)
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against out-of-bounds threads
    if (i < rows && j < rows) {
        double sum_squares = 0.0;

        // Loop over the vector dimensions
        for (int k = 0; k < cols; k++) {
            double diff = matrix[i * cols + k] - matrix[j * cols + k];
            sum_squares += diff * diff;
        }

        // Store the result back in the flat 1D array
        distance_matrix[i * rows + j] = sum_squares;
    }
}

// Host helper function to handle allocation and kernel launch
extern "C"  uint64_t* calculate_distance_matrix_cuda(uint64_t *_h_matrix,unsigned long int size) {
    int rows = (int)sqrt(size);
    int cols = rows; // Matching your assumption that rows == cols
    double *h_matrix = (double *)_h_matrix;
    size_t matrix_bytes = sizeof(double) * size;
    size_t dist_matrix_bytes = sizeof(double) * rows * rows;

    // 1. Allocate device memory
    double *d_matrix, *d_distance_matrix;
    cudaMalloc((void**)&d_matrix, matrix_bytes);
    cudaMalloc((void**)&d_distance_matrix, dist_matrix_bytes);

    // 2. Copy input matrix from host to device
    cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice);

    // 3. Define execution configuration (2D grid of blocks)
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 4. Launch the CUDA kernel
    distance_matrix_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_distance_matrix, rows, cols);

    // Wait for the GPU to finish and check for errors
    cudaDeviceSynchronize();

    // 5. Allocate host memory for the result and copy it back
    double *h_distance_matrix = (double *)malloc(dist_matrix_bytes);
    cudaMemcpy(h_distance_matrix, d_distance_matrix, dist_matrix_bytes, cudaMemcpyDeviceToHost);

    // 6. Clean up device memory
    cudaFree(d_matrix);
    cudaFree(d_distance_matrix);

    return (uint64_t *)h_distance_matrix;
}
//END