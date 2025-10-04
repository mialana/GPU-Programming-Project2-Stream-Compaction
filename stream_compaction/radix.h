#pragma once

#include "common.h"

namespace StreamCompaction
{
namespace Radix
{
StreamCompaction::Common::PerformanceTimer& timer();

__device__ __host__ int _isolateBit(const int num, const int tgtBit);

__global__ void _split(int n, int* data, int* notLSB, const int bit);

__global__ void _computeScatterIndices(int n, int* odata, const int* idata, const int* scan, const int tgtBit);

template<typename T>
__global__ void _scatter(int n, T* odata, const T* idata, const int* addresses);

void sort(int n, int* dev_dataA, int* dev_dataB, int* dev_blockSums, int* dev_indices, const int maxBitLength, const int blockSize);

void sortWrapper(
    int n, int* odata, const int* idata, const int maxBitLength, const int blockSize);

}  // namespace Radix
}  // namespace StreamCompaction
