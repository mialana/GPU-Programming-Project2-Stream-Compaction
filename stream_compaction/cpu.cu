#include "cpu.h"

#include "common.h"

namespace StreamCompaction
{
namespace CPU
{
using StreamCompaction::Common::PerformanceTimer;

PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}

/**
 * CPU scan (prefix sum).
 * For performance analysis, this is supposed to be a simple for loop.
 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan
 * in this function first.
 */
void scan(int n, int* odata, const int* idata)
{
    timer().startCpuTimer();
    // TODO

    odata[0] = 0;             // identity is 0

    int prev_sum = idata[0];  // save prev sum for access ease
    for (int j = 1; j < n + 1; j++)
    {
        odata[j] = prev_sum;
        prev_sum += idata[j];
    }

    timer().endCpuTimer();
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int* odata, const int* idata)
{
    timer().startCpuTimer();

    int outIndex = 0; // pointer to current progress in out array

    for (int i = 0; i < n; i++)
    {
        int inVal = idata[i];
        if (inVal != 0)
        {
            odata[outIndex] = inVal;
            outIndex++;
        }
    }

    timer().endCpuTimer();
    return outIndex;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int* odata, const int* idata)
{
    timer().startCpuTimer();
    // TODO
    timer().endCpuTimer();
    return -1;
}
}  // namespace CPU
}  // namespace StreamCompaction
