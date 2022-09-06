//
// Created by Dylan on 9/4/2022.
//

#ifndef DYLANN_CUCONCAT_CUH
#define DYLANN_CUCONCAT_CUH

#include "cuTensorOps.cuh"

namespace dylann {
    
    cuTensorBase* concatChannelOp(cuTensorBase** Xs, int inputCount, cuTensorBase* Y);
    
    void concatChannelOpGrads(cuTensorBase* Y, cuTensorBase** Xs, int inputCount);
    
} // dylann

#endif //DYLANN_CUCONCAT_CUH
