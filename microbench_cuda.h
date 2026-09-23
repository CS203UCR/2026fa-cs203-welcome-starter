__global__ void baseline_double_cuda_init(double *array, int size);
__global__ void baseline_double_cuda_kernel(double *_output, double *_vector, unsigned long  size);
__global__ void baseline_float_cuda(float *d_array, float *d_result, int size);
