//
// Created by Dylan on 9/4/2022.
//

#ifndef DYLANN_CUCONCAT_CUH
#define DYLANN_CUCONCAT_CUH

#include "cuTensorOps.cuh"
#include "cuTensorOpGrads.cuh"

namespace dylann {
    
    cuTensorBase* concatChannelOp(cuTensorBase** Xs, int inputCount, cuTensorBase* Y);
    
    void concatChannelOpGrads(cuTensorBase* Y, cuTensorBase** Xs, int inputCount);
    
    struct GRAD_CONCAT_CHANNEL : public GradTracker {
        cuTensorBase** Xs;
        int inputCount;
        cuTensorBase* Y;
        
        GRAD_CONCAT_CHANNEL(cuTensorBase** Xs, int inputCount, cuTensorBase* Y) : Xs(Xs), inputCount(inputCount), Y(Y) {}
    
        void backwardCalc(cuTensorBase *current) override;
    };
    
} // dylann

#endif //DYLANN_CUCONCAT_CUH
