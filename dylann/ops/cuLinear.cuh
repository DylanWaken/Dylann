//
// Created by Dylan on 8/25/2022.
//

#ifndef DYLANN_CULINEAR_CUH
#define DYLANN_CULINEAR_CUH

#include "../ops/cuTensorOpGrads.cuh"

namespace dylann {
    
    // Y = W * X + b
    cuTensorBase *linearOp(cuTensorBase* W, cuTensorBase* B, cuTensorBase* X, cuTensorBase* Y);
    
    struct GRAD_LINEAR : public GradTracker{
        cuTensorBase* W;
        cuTensorBase* B;
        cuTensorBase* X;
        explicit GRAD_LINEAR(cuTensorBase* W, cuTensorBase* B, cuTensorBase* X) : W(W), B(B), X(X){}
        
        //∂C/∂X = ∂C/∂y * ∂y/∂X = W^T * ∂C/∂y
        //∂C/∂W = ∂C/∂y * ∂y/∂W = ∂C/∂y * X^T
        //∂C/∂B = ∂C/∂y * ∂y/∂B = ∂C/∂y
        void backwardCalc(cuTensorBase* current) override;
    };
}

#endif //DYLANN_CULINEAR_CUH
