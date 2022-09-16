//
// Created by Dylan on 9/4/2022.
//

#ifndef DYLANN_CUPOOL_CUH
#define DYLANN_CUPOOL_CUH

#include "cuTensorOps.cuh"

namespace dylann {
    cuTensorBase *maxPoolOp(cuTensorBase* X, cuTensorBase* Y, int rangeH, int rangeW,
                            int padH, int padW, int strideH, int strideW, float alpha1, float alpha2);
    
    cuTensorBase *maxPoolOpGrads(cuTensorBase* X, cuTensorBase* Y,  int rangeH, int rangeW,
                                 int padH, int padW, int strideH, int strideW, float alpha1, float alpha2);
    
    
    cuTensorBase *avgPoolOp(cuTensorBase* X, cuTensorBase* Y, int rangeH, int rangeW,
                            int padH, int padW, int strideH, int strideW, float alpha1, float alpha2);
    
    cuTensorBase *avgPoolOpGrads(cuTensorBase* X, cuTensorBase* Y,  int rangeH, int rangeW,
                                 int padH, int padW, int strideH, int strideW, float alpha1, float alpha2);
    
    
    cuTensorBase *globalAvgPoolOp(cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2);
    
    cuTensorBase *globalAvgPoolOpGrads(cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2);
} // dylann

#endif //DYLANN_CUPOOL_CUH
