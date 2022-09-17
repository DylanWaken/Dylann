//
// Created by Dylan on 9/17/2022.
//

#ifndef DYLANN_LOSS_CUH
#define DYLANN_LOSS_CUH

#include "../tensor/cuTensor.cuh"
#include "../ops/cuReduce.cuh"

namespace dylann {
    
    struct Loss {
    public:
        cuTensorBase* calcBuf = nullptr;
        cuTensorBase* lossVal = nullptr;
        void* lossHost = nullptr;
        
        virtual float loss(cuTensorBase* pred, cuTensorBase* target) = 0;
        
        virtual cuTensorBase* backward(cuTensorBase* pred, cuTensorBase* target) = 0;
    };
    
    struct MSE : public Loss {
    public:
        float loss(cuTensorBase* pred, cuTensorBase* target) override;
        
        cuTensorBase* backward(cuTensorBase* pred, cuTensorBase* target) override;
    };
    
    struct CrossEntropy : public Loss {
    public:
        float loss(cuTensorBase* pred, cuTensorBase* target) override;
        
        cuTensorBase* backward(cuTensorBase* pred, cuTensorBase* target) override;
    };
    
} // dylann

#endif //DYLANN_LOSS_CUH
