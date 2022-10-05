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
        cuTensorBase* pred = nullptr;
        
        explicit Loss(cuTensorBase* pred){ this->pred = pred; }
        
        virtual float loss(cuTensorBase* target) = 0;
        
        virtual cuTensorBase* backward(cuTensorBase* target) = 0;
    };
    
    struct MSE : public Loss {
    public:
        
        explicit MSE(cuTensorBase* pred) : Loss(pred) {}
        
        float loss(cuTensorBase* target) override;
        
        cuTensorBase* backward(cuTensorBase* target) override;
    };
    
    struct CrossEntropy : public Loss {
    public:
        explicit CrossEntropy(cuTensorBase* pred) : Loss(pred) {}
        
        float loss(cuTensorBase* target) override;
        
        cuTensorBase* backward(cuTensorBase* target) override;
    };
    
} // dylann

#endif //DYLANN_LOSS_CUH
