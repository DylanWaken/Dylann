//
// Created by Dylan on 8/25/2022.
//

#ifndef DYLANN_CULINEAR_CUH
#define DYLANN_CULINEAR_CUH

#include "cuTensorOps.cuh"

namespace dylann {
    
    // B = W * A + b
    cuTensorBase *linearOp(cuTensorBase* W, cuTensorBase* B, cuTensorBase* X, cuTensorBase* Y);
    
    //∂C/∂A = ∂C/∂y * ∂y/∂A = W^T * ∂C/∂y
    //∂C/∂W = ∂C/∂y * ∂y/∂W = ∂C/∂y * A^T
    //∂C/∂B = ∂C/∂y * ∂y/∂B = ∂C/∂y
    cuTensorBase *linearOpGrads(cuTensorBase* W, cuTensorBase* B, cuTensorBase* X, cuTensorBase* Y);
}

#endif //DYLANN_CULINEAR_CUH
