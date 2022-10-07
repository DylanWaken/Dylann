//
// Created by Dylan on 9/7/2022.
//

#ifndef DYLANN_CUDROPOUT_CUH
#define DYLANN_CUDROPOUT_CUH

#include "cuTensorops.cuh"

namespace dylann {
    
    cuTensorBase* dropoutOp(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* mask, float b);
    
    cuTensorBase* dropoutOpGrads(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* mask, float b);
    
} // dylann

#endif //DYLANN_CUDROPOUT_CUH
