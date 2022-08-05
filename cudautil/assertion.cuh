//
// Created by Dylan on 8/5/2022.
//

#ifndef DYLANN_ASSERTION_CUH
#define DYLANN_ASSERTION_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include "../io/logging/LogUtils.cuh"
#include <cassert>

using namespace io;

void assertCuda(const char *file, int line);

#endif //DYLANN_ASSERTION_CUH
