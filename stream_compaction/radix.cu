#include "radix.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"
#include "shared.h"

using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& StreamCompaction::Radix::timer()
{
    static PerformanceTimer timer;
    return timer;
}

__device__ __host__ int StreamCompaction::Radix::_isolateBit(const int num, const int tgtBit)
{
    return (num >> tgtBit) & 1;
}

__global__ void StreamCompaction::Radix::_split(int n, int* data, int* notBit, const int tgtBit)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    notBit[index] = _isolateBit(data[index], tgtBit) ^ 1;  // not(target bit)
}

__global__ void StreamCompaction::Radix::_computeScatterIndices(
    int n, int* indices, const int* idata, const int* scan, const int tgtBit)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    __shared__ int totalFalses;
    if (threadIdx.x == 0)
    {
        totalFalses = (_isolateBit(idata[n - 1], tgtBit) ^ 1) + scan[n - 1];
    }

    __syncthreads();  // wait for totalFalses

    // if value is 1, we shift right by total falses minus falses before current index
    // if value is 0, we set to position based on how many other falses / 0s come before it
    indices[index] = _isolateBit(idata[index], tgtBit) ? index + (totalFalses - scan[index])
                                                       : scan[index];
}

template<typename T>
__global__ void StreamCompaction::Radix::_scatter(int n,
                                                  T* odata,
                                                  const T* idata,
                                                  const int* indices)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    int address = indices[index];
    odata[address] = idata[index];  // Scatter the value to its new position
}

void StreamCompaction::Radix::sort(
    int n, int* dev_dataA, int* dev_dataB, int* dev_blockSums, int* dev_indices, const int maxBitLength, const int blockSize)
{

    for (int tgtBit = 0; tgtBit < maxBitLength; tgtBit++)
    {
        unsigned blocks = divup(n, blockSize);

        // Split data into 0s and 1s based on the target bit
        _split<<<blocks, blockSize>>>(n, dev_dataA, dev_dataB, tgtBit);

        // Perform scan on the split results
        Shared::scan(n, dev_dataB, dev_dataB, dev_blockSums, blockSize);

        // Scatter data based on the split results
        _computeScatterIndices<<<blocks, blockSize>>>(n,
                                                      dev_indices,
                                                      dev_dataA,
                                                      dev_dataB,
                                                      tgtBit);

        _scatter<<<blocks, blockSize>>>(n, dev_dataB, dev_dataA, dev_indices);

        // Swap buffers (ping-pong)
        int* temp = dev_dataA;
        dev_dataA = dev_dataB;
        dev_dataB = temp;
    }
}

void StreamCompaction::Radix::sortWrapper(
    int n, int* odata, const int* idata, const int maxBitLength, const int blockSize)
{
    const unsigned numLayers = ilog2ceil(n);
    const unsigned long long paddedN = 1 << ilog2ceil(n);
    const unsigned blockSums = divup(paddedN, 2 * blockSize);

    // Allocate device memory for input/output data and scan
    int* dev_dataA;
    int* dev_dataB;
    int* dev_blockSums;
    int* dev_indices;

    cudaMalloc((void**)&dev_dataA, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device idata array failed.");

    cudaMalloc((void**)&dev_dataB, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for device odata array failed.");

    cudaMalloc((void**)&dev_blockSums, sizeof(int) * blockSums);
    checkCUDAError("CUDA malloc for device odata array failed.");

    cudaMalloc((void**)&dev_indices, sizeof(int) * n);
    checkCUDAError("CUDA malloc for device indices array failed.");

    // Copy input data to device
    cudaMemcpy(dev_dataA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from host data to device array failed.");


    bool usingTimer = false;
    if (!timer().gpu_timer_started)
    {
        timer().startGpuTimer();
        usingTimer = true;
    }

    StreamCompaction::Radix::sort(n, dev_dataA, dev_dataB, dev_blockSums, dev_indices, maxBitLength, blockSize);


    if (usingTimer)
    {
        timer().endGpuTimer();
    }

    // Copy sorted data back to host
    cudaMemcpy(odata, dev_dataA, sizeof(int) * n, cudaMemcpyDeviceToHost);
    checkCUDAError("Memory copy from device array to host data failed.");

    // Free device memory
    cudaFree(dev_dataA);
    cudaFree(dev_dataB);
    cudaFree(dev_blockSums);
    cudaFree(dev_indices);
}