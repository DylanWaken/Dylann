//
// Created by Dylan on 10/4/2022.
//

#ifndef DYLANN_OPTIMPL_CUH
#define DYLANN_OPTIMPL_CUH

#include "../tensor/cuTensorBase.cuh"
#include "../serial/AutoGrad.cuh"

namespace dylann {
    void momentumApply(cuTensorBase* X, cuTensorBase* m, float LEARNING_RATE, float BETA,
                       float L2, bool isWeight);
} // dylann

#endif //DYLANN_OPTIMPL_CUH
