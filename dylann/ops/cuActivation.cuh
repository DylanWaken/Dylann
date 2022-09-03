//
// Created by Dylan on 9/2/2022.
//

#ifndef DYLANN_CUACTIVATION_CUH
#define DYLANN_CUACTIVATION_CUH

#include "cuTensorOpGrads.cuh"
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
    
    
    struct GRAD_RELU : public GradTracker{
    public:
        cuTensorBase* X;
        explicit GRAD_RELU(cuTensorBase* X) : X(X){}
        
        void backwardCalc(dylann::cuTensorBase *Y) override;
    };
    
    struct GRAD_SIGMOID : public GradTracker{
    public:
        cuTensorBase* X;
        explicit GRAD_SIGMOID(cuTensorBase* X) : X(X){}
        
        void backwardCalc(dylann::cuTensorBase *Y) override;
    };
    
    struct GRAD_TANH : public GradTracker{
    public:
        cuTensorBase* X;
        explicit GRAD_TANH(cuTensorBase* X) : X(X){}
        
        void backwardCalc(dylann::cuTensorBase *Y) override;
    };
    
    struct GRAD_ELU : public GradTracker{
    public:
        cuTensorBase* X;
        float alpha;
        GRAD_ELU(cuTensorBase* X, float alpha) : X(X), alpha(alpha){}
        
        void backwardCalc(dylann::cuTensorBase *Y) override;
    };
    
    struct GRAD_SWISH : public GradTracker{
    public:
        cuTensorBase* X;
        float beta;
        GRAD_SWISH(cuTensorBase* X, float beta) : X(X), beta(beta){}
        
        void backwardCalc(dylann::cuTensorBase *Y) override;
    };
    
    struct GRAD_CLIPPED_RELU : public GradTracker{
    public:
        cuTensorBase* X;
        float threshold;
        GRAD_CLIPPED_RELU(cuTensorBase* X, float threshold) : X(X), threshold(threshold){}
        
        void backwardCalc(dylann::cuTensorBase *Y) override;
    };
}

#endif //DYLANN_CUACTIVATION_CUH
