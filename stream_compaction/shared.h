#pragma once

#include "common.h"

namespace StreamCompaction
{
namespace Shared
{
StreamCompaction::Common::PerformanceTimer& timer();

void scan(int n, int* dev_idata, int* dev_odata, int* dev_blockSums, const int blockSize);

void scanWrapper(int n, int* odata, const int* idata);

int compact(int n, int* odata, const int* idata);
}  // namespace Shared
}  // namespace StreamCompaction
