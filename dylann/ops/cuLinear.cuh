//
// Created by Dylan on 8/25/2022.
//

#ifndef DYLANN_CULINEAR_CUH
#define DYLANN_CULINEAR_CUH

#include "../ops/opRegistry.cuh"

namespace dylann {
    cuTensorBase *linearOp(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y);
}

#endif //DYLANN_CULINEAR_CUH
