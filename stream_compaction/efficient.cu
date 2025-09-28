#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction
{
namespace Efficient
{
using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}

// iter = d
__global__ void kernel_efficientUpSweep(const unsigned long long paddedN,
                                        const int iter,
                                        const int stride,
                                        const int prevStride,
                                        int* scan)
{
    int strideIdx = blockIdx.x * blockDim.x + threadIdx.x;  // 0, 1, 2, 3... (like normal)
                                                            // but this is not target elem index

    unsigned long long strideStart = strideIdx * stride; // index where this stride starts

    // last index in stride. accumulated value of stride always goes here
    unsigned long long accumulatorIdx = strideStart + stride - 1;

    if (accumulatorIdx >= paddedN)
    {
        return;
    }

    int accumulator = scan[accumulatorIdx]; // pre-fetch accumulator's value

    // this new stride has swallowed two strides total
    // siblingIdx is the index of the other stride that now no longer exists
    unsigned long long siblingIdx = strideStart + prevStride - 1; // doesn't depend on accumulator

    scan[accumulatorIdx] = accumulator + scan[siblingIdx];
}

__global__ void kernel_efficientDownSweep(const int n, const int iter, int* scan)
{
    int iterTarget = 1 << (iter + 1);
    int iterFactor = 1 << iter;

    unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x;
    index = index * iterTarget;

    if (index + iterTarget - 1 < n)
    {
        int leftChild = scan[index + iterFactor - 1];
        scan[index + iterFactor - 1] = scan[index + iterTarget - 1];
        scan[index + iterTarget - 1] += leftChild;
    }
}

/*
    the inner operation of scan without timers and allocation.
    note: dev_scan should be pre-allocated to the padded power of two size
*/
void scan(int n, int* dev_scan)
{
    // unsigned long long numLayers = ilog2ceil(n);
    int numLayers = ilog2ceil(n);
    unsigned long long paddedN = 1 << numLayers;  // pad to nearest power of 2

    int prevStride = 1;
    int stride = 2;  // essentially the amount of indices that are accumulated into 1 at this iter
    for (int iter = 0; iter < numLayers; iter++)
    {
        // paddedN >> (iter + 1) == paddedN / (iter + 2) = the number of active threads in this iter
        int blocks = divup(paddedN >> (iter + 1), BLOCK_SIZE);
        kernel_efficientUpSweep<<<blocks, BLOCK_SIZE>>>(paddedN, iter, stride, prevStride, dev_scan);
        checkCUDAError("Perform Work-Efficient Scan Up Sweep Iteration CUDA kernel failed.");

        prevStride = stride;
        stride = stride << 1;  // 1, 2, 4, 8, 16, ...
    }

    Common::kernel_setDeviceArrayValue<<<1, 1>>>(dev_scan, paddedN - 1, 0);

    for (int i = numLayers - 1; i >= 0; i--)
    {
        int blocks = divup(paddedN / (1 << (i + 1)), BLOCK_SIZE);
        kernel_efficientDownSweep<<<blocks, BLOCK_SIZE>>>(paddedN, i, dev_scan);
        checkCUDAError("Perform Work-Efficient Scan Down Sweep Iteration CUDA kernel failed.");
    }
}

/************************************************************************************************ */

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scanWrapper(int n, int* odata, const int* idata)
{
    unsigned long long numLayers = ilog2ceil(n);
    unsigned long long paddedN = 1 << ilog2ceil(n);

    // create two device arrays
    int* dev_scan;

    cudaMalloc((void**)&dev_scan, sizeof(int) * paddedN);
    checkCUDAError("CUDA malloc for scan array failed.");

    cudaMemcpy(dev_scan, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from input data to scan array failed.");

    cudaDeviceSynchronize();

    bool usingTimer = false;
    if (!timer().gpu_timer_started)  // added in order to call `scan` from other functions.
    {
        timer().startGpuTimer();
        usingTimer = true;
    }

    scan(n, dev_scan);

    if (usingTimer)
    {
        timer().endGpuTimer();
    }

    cudaMemcpy(odata, dev_scan, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(dev_scan);  // can't forget memory leaks!
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int* odata, const int* idata)
{
    // TODO: these arrays are unnecessary. will optimize soon.

    // create device arrays
    int* dev_idata;
    int* dev_odata;

    int* dev_bools;
    int* dev_indices;

    cudaMalloc((void**)&dev_idata, sizeof(int) * n);
    checkCUDAError("CUDA malloc for idata array failed.");

    cudaMalloc((void**)&dev_odata, sizeof(int) * n);
    checkCUDAError("CUDA malloc for odata array failed.");

    cudaMalloc((void**)&dev_bools, sizeof(int) * n);
    checkCUDAError("CUDA malloc for bools array failed.");

    cudaMalloc((void**)&dev_indices, sizeof(int) * n);
    checkCUDAError("CUDA malloc for indices array failed.");

    cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from input data to idata array failed.");
    cudaMemcpy(dev_bools, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from output data to odata array failed.");

    cudaDeviceSynchronize();

    int* indices = new int[n];  // create cpu side indices array
    int* bools = new int[n];

    timer().startGpuTimer();

    int blocks = divup(n, BLOCK_SIZE);

    // reuse dev_idata for bools
    Common::kernMapToBoolean<<<blocks, BLOCK_SIZE>>>(n, dev_bools, dev_idata);

    cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
    checkCUDAError("Memory copy from device bools to indices array failed.");

    scanWrapper(n, indices, bools);

    cudaMemcpy(dev_indices, indices, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from indices to device indices array failed.");

    Common::kernScatter<<<blocks, BLOCK_SIZE>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

    timer().endGpuTimer();

    cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(dev_idata);
    cudaFree(dev_odata);
    cudaFree(dev_bools);
    cudaFree(dev_indices);

    return indices[n - 1] + bools[n - 1];
}
}  // namespace Efficient
}  // namespace StreamCompaction
