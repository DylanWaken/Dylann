//
// Created by Dylan on 9/4/2022.
//

#ifndef DYLANN_CUBATCHNORM_CUH
#define DYLANN_CUBATCHNORM_CUH

#include "cuTensorOps.cuh"
#include "cuTensorOpGrads.cuh"

namespace dylann {
    
    cuTensorBase* batchnormOp(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                              cuTensorBase* gamma, cuTensorBase* beta, float eps, float expAvgFactor);
    
    cuTensorBase* batchnormInferOp(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                                   cuTensorBase* gamma, cuTensorBase* beta, float eps);
    
    cuTensorBase* batchnormOpGrads(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                                   cuTensorBase* gamma, cuTensorBase* beta, float eps, float expAvgFactor);
    
    struct GRAD_BATCHNORM : public GradTracker{
        cuTensorBase* X;
        cuTensorBase* runningMean;
        cuTensorBase* runningVar;
        cuTensorBase* gamma;
        cuTensorBase* beta;
        float eps;
        float expAvgFactor;
        explicit GRAD_BATCHNORM(cuTensorBase* X, cuTensorBase* runningMean, cuTensorBase* runningVar,
                                cuTensorBase* gamma, cuTensorBase* beta, float eps, float expAvgFactor) : X(X),
                                runningMean(runningMean), runningVar(runningVar),
                                gamma(gamma), beta(beta),
                                eps(eps), expAvgFactor(expAvgFactor){}
        
        void backwardCalc(cuTensorBase* Y) override;
    };
} // dylann

#endif //DYLANN_CUBATCHNORM_CUH
