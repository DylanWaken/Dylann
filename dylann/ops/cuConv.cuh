//
// Created by Dylan on 8/31/2022.
//

#ifndef DYLANN_CUCONV_CUH
#define DYLANN_CUCONV_CUH

#include "../tensor/cuTensorBase.cuh"
#include "cuTensorOpGrads.cuh"

namespace dylann{
    cuTensorBase* conv2dOp(cuTensorBase* X, cuTensorBase* W, cuTensorBase* B, cuTensorBase* Y,
                           int padH, int padW, int strideH, int strideW, int dilationH, int dilationW);
    
    
    cuTensorBase* conv2dActiveOp(cuTensorBase* X, cuTensorBase* W, cuTensorBase* B, cuTensorBase* Y,
                                 int padH, int padW, int strideH, int strideW, int dilationH, int dilationW,
                                 cudnnActivationMode_t mode, float coef);
    
    struct GRAD_CONV2D : public GradTracker{
        cuTensorBase* X;
        cuTensorBase* W;
        cuTensorBase* B;
        int padH, padW, strideH, strideW, dilationH, dilationW;
        explicit GRAD_CONV2D(cuTensorBase* X, cuTensorBase* W, cuTensorBase* B,
                             int padH, int padW, int strideH, int strideW, int dilationH, int dilationW) :
                X(X), W(W), B(B), padH(padH), padW(padW),
                strideH(strideH), strideW(strideW),
                dilationH(dilationH), dilationW(dilationW){}
        
        //∂C/∂X = ∂C/∂y * ∂y/∂X = W^T * ∂C/∂y
        //∂C/∂W = ∂C/∂y * ∂y/∂W = ∂C/∂y * X^T
        //∂C/∂B = ∂C/∂y * ∂y/∂B = ∂C/∂y
        void backwardCalc(cuTensorBase* Y) override;
    };
}


#endif //DYLANN_CUCONV_CUH
