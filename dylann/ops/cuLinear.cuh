//
// Created by Dylan on 8/25/2022.
//

#ifndef DYLANN_CULINEAR_CUH
#define DYLANN_CULINEAR_CUH

#include "../ops/cuTensorOpGrads.cuh"

namespace dylann {
    
    // B = W * A + b
    cuTensorBase *linearOp(cuTensorBase* W, cuTensorBase* B, cuTensorBase* X, cuTensorBase* Y);
    cuTensorBase *linearOpGrads(cuTensorBase* W, cuTensorBase* B, cuTensorBase* X, cuTensorBase* Y);
    
    struct GRAD_LINEAR : public GradTracker{
        cuTensorBase* W;
        cuTensorBase* B;
        cuTensorBase* X;
        explicit GRAD_LINEAR(cuTensorBase* W, cuTensorBase* B, cuTensorBase* X) : W(W), B(B), X(X){}
        
        //∂C/∂A = ∂C/∂y * ∂y/∂A = W^T * ∂C/∂y
        //∂C/∂W = ∂C/∂y * ∂y/∂W = ∂C/∂y * A^T
        //∂C/∂B = ∂C/∂y * ∂y/∂B = ∂C/∂y
        void backwardCalc(cuTensorBase* Y) override;
    };
}

#endif //DYLANN_CULINEAR_CUH
