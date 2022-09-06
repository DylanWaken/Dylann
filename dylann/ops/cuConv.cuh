//
// Created by Dylan on 8/31/2022.
//

#ifndef DYLANN_CUCONV_CUH
#define DYLANN_CUCONV_CUH

#include "cuTensorOps.cuh"

namespace dylann{
    cuTensorBase* conv2dOp(cuTensorBase* X, cuTensorBase* W, cuTensorBase* B, cuTensorBase* Y,
                           int padH, int padW, int strideH, int strideW, int dilationH, int dilationW);
    cuTensorBase* conv2dOpGrads(cuTensorBase* X, cuTensorBase* W, cuTensorBase* B, cuTensorBase* Y,
                                int padH, int padW, int strideH, int strideW, int dilationH, int dilationW);
    
    
    cuTensorBase* conv2dActiveOp(cuTensorBase* X, cuTensorBase* W, cuTensorBase* B, cuTensorBase* Y,
                                 int padH, int padW, int strideH, int strideW, int dilationH, int dilationW,
                                 cudnnActivationMode_t mode, float coef);
    
}


#endif //DYLANN_CUCONV_CUH
