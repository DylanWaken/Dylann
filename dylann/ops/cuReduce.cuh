//
// Created by Dylan on 9/2/2022.
//

#ifndef DYLANN_CUREDUCE_CUH
#define DYLANN_CUREDUCE_CUH

#include "cuTensorOps.cuh"

#define ONE_VEC_BUF_SIZE 160000

namespace dylann{
    
    extern void* oneVec10KF;

    cuTensorBase* reduceOp(cuTensorBase* X, cuTensorBase* Y, int step);
    
    cuTensorBase* softmaxOp(cuTensorBase* X, cuTensorBase* Y, int step);
    cuTensorBase* softmaxOpGrads(cuTensorBase* X, cuTensorBase* Y, int step);
    
    cuTensorBase* softmaxLogOp(cuTensorBase* X, cuTensorBase* Y, int step);
    cuTensorBase* softmaxLogOpGrads(cuTensorBase* X, cuTensorBase* Y, int step);
    
    //cross entropy loss operation
    cuTensorBase* softmaxCEOp(cuTensorBase* X, cuTensorBase* Y, int step);
    cuTensorBase* softmaxCEOpGrads(cuTensorBase* X, cuTensorBase* Y, int step);

}

#endif //DYLANN_CUREDUCE_CUH
