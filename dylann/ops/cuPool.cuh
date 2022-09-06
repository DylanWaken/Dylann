//
// Created by Dylan on 9/4/2022.
//

#ifndef DYLANN_CUPOOL_CUH
#define DYLANN_CUPOOL_CUH

#include "cuTensorOps.cuh"

namespace dylann {
    cuTensorBase *maxPoolOp(cuTensorBase* X, cuTensorBase* Y, int rangeH, int rangeW,
                            int padH, int padW, int strideH, int strideW);
    
    cuTensorBase *maxPoolOpGrads(cuTensorBase* X, cuTensorBase* Y,  int rangeH, int rangeW,
                                 int padH, int padW, int strideH, int strideW);
    
    
    cuTensorBase *avgPoolOp(cuTensorBase* X, cuTensorBase* Y, int rangeH, int rangeW,
                            int padH, int padW, int strideH, int strideW);
    
    cuTensorBase *avgPoolOpGrads(cuTensorBase* X, cuTensorBase* Y,  int rangeH, int rangeW,
                                 int padH, int padW, int strideH, int strideW);
    
    
    cuTensorBase *globalAvgPoolOp(cuTensorBase* X, cuTensorBase* Y);
    
    cuTensorBase *globalAvgPoolOpGrads(cuTensorBase* X, cuTensorBase* Y);
} // dylann

#endif //DYLANN_CUPOOL_CUH
