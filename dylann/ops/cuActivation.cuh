//
// Created by Dylan on 9/2/2022.
//

#ifndef DYLANN_CUACTIVATION_CUH
#define DYLANN_CUACTIVATION_CUH

#include "cuTensorOps.cuh"

namespace dylann{
    
    extern cudnnActivationDescriptor_t reluDescG;
    extern cudnnActivationDescriptor_t sigmoidDescG;
    extern cudnnActivationDescriptor_t tanhDescG;

    
    cuTensorBase* reluOp(cuTensorBase* X);
    cuTensorBase* reluOp(cuTensorBase* X, cuTensorBase* Y);
    cuTensorBase* reluOpGrads(cuTensorBase* X, cuTensorBase* Y);
    
    cuTensorBase* sigmoidOp(cuTensorBase* X);
    cuTensorBase* sigmoidOp(cuTensorBase* X, cuTensorBase* Y);
    cuTensorBase* sigmoidOpGrads(cuTensorBase* X, cuTensorBase* Y);
    
    cuTensorBase* tanhOp(cuTensorBase* X);
    cuTensorBase* tanhOp(cuTensorBase* X, cuTensorBase* Y);
    cuTensorBase* tanhOpGrads(cuTensorBase* X, cuTensorBase* Y);
    
    cuTensorBase* eluOp(cuTensorBase* X, float alpha);
    cuTensorBase* eluOp(cuTensorBase* X, cuTensorBase* Y, float alpha);
    cuTensorBase* eluOpGrads(cuTensorBase* X, cuTensorBase* Y, float alpha);
    
    cuTensorBase* swishOp(cuTensorBase* X, float beta);
    cuTensorBase* swishOp(cuTensorBase* X, cuTensorBase* Y, float beta);
    cuTensorBase* swishOpGrads(cuTensorBase* X, cuTensorBase* Y, float beta);
    
    cuTensorBase* clippedReluOp(cuTensorBase* X, float threshold);
    cuTensorBase* clippedReluOp(cuTensorBase* X, cuTensorBase* Y, float threshold);
    cuTensorBase* clippedReluOpGrads(cuTensorBase* X, cuTensorBase* Y, float threshold);
}

#endif //DYLANN_CUACTIVATION_CUH
