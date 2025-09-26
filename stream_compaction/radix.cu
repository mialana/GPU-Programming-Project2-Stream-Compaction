#include "radix.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"

namespace StreamCompaction
{
namespace Radix
{
using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}

__device__ __host__ int _isolateBit(const int num, const int tgtBit)
{
    return (num >> tgtBit) & 1;
}

__global__ void _split(int n, int* data, int* notBit, const int tgtBit)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    notBit[index] = _isolateBit(data[index], tgtBit) ^ 1;  // not(target bit)
}

__global__ void _scatter(int n, int* data, const int* scan, const int tgtBit)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    __shared__ int totalFalses;
    if (threadIdx.x == 0)
    {
        totalFalses = (_isolateBit(data[n - 1], tgtBit) ^ 1) + scan[n - 1];
    }

    int savedVal = data[index];
    __syncthreads(); // wait for totalFalses and savedVal

    int address = (_isolateBit(savedVal, tgtBit)) ? (scan[index])
                                                 : index - scan[index] + totalFalses;

    __syncthreads();

    data[address] = savedVal;
}

void sort(int n, int* odata, const int* idata, const int maxBitLength)
{
    // create device arrays
    int* dev_arr1;
    int* dev_arr2;

    cudaMalloc((void**)&dev_arr1, sizeof(int) * n);
    checkCUDAError("CUDA malloc for 1st array failed.");

    cudaMalloc((void**)&dev_arr2, sizeof(int) * n);
    checkCUDAError("CUDA malloc for 2nd array failed.");

    cudaMemcpy(dev_arr1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from host idata to device array failed.");

    for (int tgtBit = 0; tgtBit < maxBitLength; tgtBit++)
    {
        unsigned blocks = divup(n, BLOCK_SIZE);
        _split<<<blocks, BLOCK_SIZE>>>(n, dev_arr1, dev_arr2, tgtBit);

        Efficient::scanHelper(ilog2ceil(n), 1 << ilog2ceil(n), dev_arr2);

        _scatter<<<blocks, BLOCK_SIZE>>>(n, dev_arr1, dev_arr2, tgtBit);
    }

    cudaMemcpy(odata, dev_arr1, sizeof(int) * n, cudaMemcpyDeviceToHost);
    checkCUDAError("Memory copy from device array to host odata failed.");

    cudaFree(dev_arr1);
    cudaFree(dev_arr2);
}

}  // namespace Radix
}  // namespace StreamCompaction
