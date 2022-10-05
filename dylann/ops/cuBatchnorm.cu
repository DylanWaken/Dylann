//
// Created by Dylan on 9/4/2022.
//

#include "cuBatchnorm.cuh"

namespace dylann {
    cuTensorBase* batchnormOp(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                              cuTensorBase* gamma, cuTensorBase* beta, float eps, float expAvgFactor, float alpha1, float alpha2) {
        cudaSetDevice(X->data->deviceID);
        
//        float a = 1.0f, b = 0.0f;
        
        //in this function, to save space, we use the grads for runningMean and runningVar as temp storage
        checkCUDNN(cudnnBatchNormalizationForwardTraining(
                cudnnHdlG,
                CUDNN_BATCHNORM_PER_ACTIVATION,
                &alpha1,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data,
                Y->desc.cudnnDesc,
                Y->data->data,
                gamma->desc.cudnnDesc,
                gamma->data->data,
                beta->data->data,
                expAvgFactor,
                runningMean->data->data,
                runningVar->data->data,
                eps,
                runningMean->grad->data,
                runningVar->grad->data
                ))
                
        return Y;
    }
    
    cuTensorBase* batchnormInferOp(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                                   cuTensorBase* gamma, cuTensorBase* beta, float eps, float alpha1, float alpha2) {
        cudaSetDevice(X->data->deviceID);
        
//        float a = 1.0f, b = 0.0f;
        
        checkCUDNN(cudnnBatchNormalizationForwardInference(
                cudnnHdlG,
                CUDNN_BATCHNORM_PER_ACTIVATION,
                &alpha1,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data,
                Y->desc.cudnnDesc,
                Y->data->data,
                gamma->desc.cudnnDesc,
                gamma->data->data,
                beta->data->data,
                runningMean->data->data,
                runningVar->data->data,
                eps
                ))
                
        return Y;
    }
    
    cuTensorBase* batchnormOpGrads(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                                   cuTensorBase* gamma, cuTensorBase* beta, float eps, float expAvgFactor, float alpha1, float alpha2) {
        cudaSetDevice(X->data->deviceID);
        
//        float a = 1.0f, b = 1.0f;
        
        checkCUDNN(cudnnBatchNormalizationBackward(
                cudnnHdlG,
                CUDNN_BATCHNORM_PER_ACTIVATION,
                &alpha1,
                &alpha2,
                &alpha1,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data,
                Y->desc.cudnnDesc,
                Y->grad->data,
                X->desc.cudnnDesc,
                X->grad->data,
                gamma->desc.cudnnDesc,
                gamma->data->data,
                gamma->grad->data,
                beta->grad->data,
                eps,
                nullptr,//runningMean->grad->data,
                nullptr//runningVar->grad->data
                ))
                
        return X;
    }
    
    cuTensorBase* batchnorm2dOp(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                                cuTensorBase* gamma, cuTensorBase* beta, float eps, float expAvgFactor, float alpha1, float alpha2){
        //in this function, to save space, we use the grads for runningMean and runningVar as temp storage
        checkCUDNN(cudnnBatchNormalizationForwardTraining(
                cudnnHdlG,
                CUDNN_BATCHNORM_SPATIAL,
                &alpha1,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data,
                Y->desc.cudnnDesc,
                Y->data->data,
                gamma->desc.cudnnDesc,
                gamma->data->data,
                beta->data->data,
                expAvgFactor,
                runningMean->data->data,
                runningVar->data->data,
                eps,
                runningMean->grad->data,
                runningVar->grad->data
        ))
    
        return Y;
    }
    
    cuTensorBase* batchnorm2dInferOp(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                                     cuTensorBase* gamma, cuTensorBase* beta, float eps, float alpha1, float alpha2){
        checkCUDNN(cudnnBatchNormalizationForwardInference(
                cudnnHdlG,
                CUDNN_BATCHNORM_SPATIAL,
                &alpha1,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data,
                Y->desc.cudnnDesc,
                Y->data->data,
                gamma->desc.cudnnDesc,
                gamma->data->data,
                beta->data->data,
                runningMean->data->data,
                runningVar->data->data,
                eps
        ))
        
        return Y;
    }
    
    cuTensorBase* batchnorm2dOpGrads(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* runningMean, cuTensorBase* runningVar,
                                     cuTensorBase* gamma, cuTensorBase* beta, float eps, float expAvgFactor, float alpha1, float alpha2){
        checkCUDNN(cudnnBatchNormalizationBackward(
                cudnnHdlG,
                CUDNN_BATCHNORM_SPATIAL,
                &alpha1,
                &alpha2,
                &alpha1,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data,
                Y->desc.cudnnDesc,
                Y->grad->data,
                X->desc.cudnnDesc,
                X->grad->data,
                gamma->desc.cudnnDesc,
                gamma->data->data,
                gamma->grad->data,
                beta->grad->data,
                eps,
                nullptr,//runningMean->grad->data,
                nullptr//runningVar->grad->data
        ))
        
        return X;
    }
} // dylann