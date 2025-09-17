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

__global__ void kernel_performEfficientScanUpSweepIteration(const int n, const int iter, int* scan)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
    {
        return;
    }
    int iterFactor = exp2f(iter);

    int iterTarget = exp2f(iter + 1);
    if (index % iterTarget == 0)
    {
        scan[index + iterTarget - 1] += scan[index + iterFactor - 1];
    }
}

__global__ void kernel_performEfficientScanDownSweepIteration(const int n, const int iter, int* scan)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
    {
        return;
    }

    int iterFactor = exp2f(iter);
    int iterTarget = exp2f(iter + 1);

    if (index % iterTarget == 0)
    {
        int leftChild = scan[index + iterFactor - 1];
        scan[index + iterFactor - 1] = scan[index + iterTarget - 1];

        scan[index + iterTarget - 1] += leftChild;
    }
}

__global__ void kernel_setFirstZero(const int n, int* scan)
{
    scan[(int)exp2f(n)-1] = 0; // round up to nearest power of two
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int* odata, const int* idata)
{
    // create two device arrays
    int* dev_scan;

    cudaMalloc((void**)&dev_scan, sizeof(int) * n);
    checkCUDAError("CUDA malloc for scan array failed.");

    cudaMemcpy(dev_scan, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    checkCUDAError("Memory copy from input data to scan array failed.");

    cudaDeviceSynchronize();

    timer().startGpuTimer();

    int blocks = divup(n, BLOCK_SIZE);

    for (int i = 0; i <= ilog2ceil(n) - 1; i++)
    {
        kernel_performEfficientScanUpSweepIteration<<<blocks, BLOCK_SIZE>>>(n, i, dev_scan);
        checkCUDAError("Perform Work-Efficient Scan Up Sweep Iteration CUDA kernel failed.");
    }

    kernel_setFirstZero<<<1, 1>>>(ilog2ceil(n), dev_scan);

    for (int i = ilog2ceil(n)-1; i >= 0; i--)
    {
        kernel_performEfficientScanDownSweepIteration<<<blocks, BLOCK_SIZE>>>(n, i, dev_scan);
        checkCUDAError("Perform Work-Efficient Scan Down Sweep Iteration CUDA kernel failed.");
    }

    timer().endGpuTimer();

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
    timer().startGpuTimer();
    // TODO
    timer().endGpuTimer();
    return -1;
}
}  // namespace Efficient
}  // namespace StreamCompaction
