//
// Created by Dylan on 9/2/2022.
//

#ifndef DYLANN_CUREDUCE_CUH
#define DYLANN_CUREDUCE_CUH

#include "cuTensorOpGrads.cuh"
#include "cuTensorOps.cuh"

#define ONE_VEC_BUF_SIZE 160000

namespace dylann{
    
    extern void* oneVec10KF;

    cuTensorBase* reduceOp(cuTensorBase* X, cuTensorBase* Y, int step);
}

#endif //DYLANN_CUREDUCE_CUH
