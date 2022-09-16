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

    
    cuTensorBase* reluOp(cuTensorBase* X , float alpha1, float alpha2);
    cuTensorBase* reluOp(cuTensorBase* X, cuTensorBase* Y , float alpha1, float alpha2);
    cuTensorBase* reluOpGrads(cuTensorBase* X, cuTensorBase* Y , float alpha1, float alpha2);
    
    cuTensorBase* sigmoidOp(cuTensorBase* X , float alpha1, float alpha2);
    cuTensorBase* sigmoidOp(cuTensorBase* X, cuTensorBase* Y , float alpha1, float alpha2);
    cuTensorBase* sigmoidOpGrads(cuTensorBase* X, cuTensorBase* Y , float alpha1, float alpha2);
    
    cuTensorBase* tanhOp(cuTensorBase* X , float alpha1, float alpha2);
    cuTensorBase* tanhOp(cuTensorBase* X, cuTensorBase* Y , float alpha1, float alpha2);
    cuTensorBase* tanhOpGrads(cuTensorBase* X, cuTensorBase* Y , float alpha1, float alpha2);
    
    cuTensorBase* eluOp(cuTensorBase* X, float alpha , float alpha1, float alpha2);
    cuTensorBase* eluOp(cuTensorBase* X, cuTensorBase* Y, float alpha , float alpha1, float alpha2);
    cuTensorBase* eluOpGrads(cuTensorBase* X, cuTensorBase* Y, float alpha , float alpha1, float alpha2);
    
    cuTensorBase* swishOp(cuTensorBase* X, float beta , float alpha1, float alpha2);
    cuTensorBase* swishOp(cuTensorBase* X, cuTensorBase* Y, float beta , float alpha1, float alpha2);
    cuTensorBase* swishOpGrads(cuTensorBase* X, cuTensorBase* Y, float beta , float alpha1, float alpha2);
    
    cuTensorBase* clippedReluOp(cuTensorBase* X, float threshold , float alpha1, float alpha2);
    cuTensorBase* clippedReluOp(cuTensorBase* X, cuTensorBase* Y, float threshold , float alpha1, float alpha2);
    cuTensorBase* clippedReluOpGrads(cuTensorBase* X, cuTensorBase* Y, float threshold , float alpha1, float alpha2);
}

#endif //DYLANN_CUACTIVATION_CUH
