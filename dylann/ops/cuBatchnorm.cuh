//
// Created by Dylan on 9/4/2022.
//

#ifndef DYLANN_CUBATCHNORM_CUH
#define DYLANN_CUBATCHNORM_CUH

#include "cuTensorOps.cuh"

namespace dylann {
    
    cuTensorBase* batchnormOp(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                              cuTensorBase* gamma, cuTensorBase* beta, float eps, float expAvgFactor, float alpha1, float alpha2);
    
    cuTensorBase* batchnormInferOp(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                                   cuTensorBase* gamma, cuTensorBase* beta, float eps, float alpha1, float alpha2);
    
    cuTensorBase* batchnormOpGrads(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                                   cuTensorBase* gamma, cuTensorBase* beta, float eps, float expAvgFactor, float alpha1, float alpha2);
} // dylann

#endif //DYLANN_CUBATCHNORM_CUH
