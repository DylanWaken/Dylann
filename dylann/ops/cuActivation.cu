//
// Created by Dylan on 9/2/2022.
//

#include "cuActivation.cuh"

namespace dylann{
    cudnnActivationDescriptor_t reluDescG;
    cudnnActivationDescriptor_t sigmoidDescG;
    cudnnActivationDescriptor_t tanhDescG;
    
    cuTensorBase* reluOp(cuTensorBase* X){
        
        if(reluDescG == nullptr){
            cudnnCreateActivationDescriptor(&reluDescG);
            cudnnSetActivationDescriptor(reluDescG, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0);
        }
        
        float alpha = 1.0f, alpha2 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                reluDescG,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data
                ))
                
        return X;
    }
    
    cuTensorBase* reluOp(cuTensorBase* X, cuTensorBase* Y){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
        
        if(reluDescG == nullptr){
            cudnnCreateActivationDescriptor(&reluDescG);
            cudnnSetActivationDescriptor(reluDescG, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0);
        }
        
        float alpha = 1.0f, alpha2 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                reluDescG,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha2,
                Y->desc.cudnnDesc,
                Y->data->data
                ))
                
        return Y;
    }
    
    cuTensorBase* reluOpGrads(cuTensorBase* X, cuTensorBase* Y){
        float alpha = 1.0f, beta = 0.0f;
    
        checkCUDNN(cudnnActivationBackward(cudnnHdlG,
                                           reluDescG,
                                           &alpha,
                                           Y->desc.cudnnDesc,
                                           Y->data->data,
                                           Y->desc.cudnnDesc,
                                           Y->grad->data,
                                           X->desc.cudnnDesc,
                                           X->data->data,
                                           &beta,
                                           X->desc.cudnnDesc,
                                           X->grad->data))
        return X;
    }
    
    
    cuTensorBase* sigmoidOp(cuTensorBase* X){
        if(reluDescG == nullptr){
            cudnnCreateActivationDescriptor(&sigmoidDescG);
            cudnnSetActivationDescriptor(sigmoidDescG, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0);
        }
        
        float alpha = 1.0f, alpha2 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                sigmoidDescG,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data
                ))
                
        return X;
    }
    
    cuTensorBase* sigmoidOp(cuTensorBase* X, cuTensorBase* Y){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
        
        if(reluDescG == nullptr){
            cudnnCreateActivationDescriptor(&sigmoidDescG);
            cudnnSetActivationDescriptor(sigmoidDescG, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0);
        }
        
        float alpha = 1.0f, alpha2 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                sigmoidDescG,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha2,
                Y->desc.cudnnDesc,
                Y->data->data
                ))
                
        return Y;
    }
    
    cuTensorBase* sigmoidOpGrads(cuTensorBase* X, cuTensorBase* Y){
        float alpha = 1.0f, beta = 0.0f;
    
        checkCUDNN(cudnnActivationBackward(cudnnHdlG,
                                           sigmoidDescG,
                                           &alpha,
                                           Y->desc.cudnnDesc,
                                           Y->data->data,
                                           Y->desc.cudnnDesc,
                                           Y->grad->data,
                                           X->desc.cudnnDesc,
                                           X->data->data,
                                           &beta,
                                           X->desc.cudnnDesc,
                                           X->grad->data))
        return X;
    }
    
    
    cuTensorBase* tanhOp(cuTensorBase* X){
        if(reluDescG == nullptr){
            cudnnCreateActivationDescriptor(&tanhDescG);
            cudnnSetActivationDescriptor(tanhDescG, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0);
        }
        
        float alpha = 1.0f, alpha2 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                tanhDescG,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data
                ))
                
        return X;
    }
    
    cuTensorBase* tanhOp(cuTensorBase* X, cuTensorBase* Y){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
        
        if(reluDescG == nullptr){
            cudnnCreateActivationDescriptor(&reluDescG);
            cudnnSetActivationDescriptor(tanhDescG, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0);
        }
        
        float alpha = 1.0f, alpha2 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                tanhDescG,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha2,
                Y->desc.cudnnDesc,
                Y->data->data
                ))
                
        return Y;
    }
    
    cuTensorBase* tanhOpGrads(cuTensorBase* X, cuTensorBase* Y){
        float alpha = 1.0f, beta = 0.0f;
    
        checkCUDNN(cudnnActivationBackward(cudnnHdlG,
                                           tanhDescG,
                                           &alpha,
                                           Y->desc.cudnnDesc,
                                           Y->data->data,
                                           Y->desc.cudnnDesc,
                                           Y->grad->data,
                                           X->desc.cudnnDesc,
                                           X->data->data,
                                           &beta,
                                           X->desc.cudnnDesc,
                                           X->grad->data))
        return X;
    }
    
    
    cuTensorBase* eluOp(cuTensorBase* X, float alpha){

        cudnnActivationDescriptor_t eluDesc;
        cudnnCreateActivationDescriptor(&eluDesc);
        cudnnSetActivationDescriptor(eluDesc, CUDNN_ACTIVATION_ELU, CUDNN_NOT_PROPAGATE_NAN, alpha);
        
        
        float alpha2 = 1.0f, alpha3 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                eluDesc,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha3,
                X->desc.cudnnDesc,
                X->data->data
                ))
    
        cudnnDestroyActivationDescriptor(eluDesc);
        return X;
    }
    
    cuTensorBase* eluOp(cuTensorBase* X, cuTensorBase* Y, float alpha){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
        
        cudnnActivationDescriptor_t eluDesc;
        cudnnCreateActivationDescriptor(&eluDesc);
        cudnnSetActivationDescriptor(eluDesc, CUDNN_ACTIVATION_ELU, CUDNN_NOT_PROPAGATE_NAN, alpha);
        
        
        float alpha2 = 1.0f, alpha3 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                eluDesc,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha3,
                Y->desc.cudnnDesc,
                Y->data->data
                ))
                
        cudnnDestroyActivationDescriptor(eluDesc);
        return Y;
    }
    
    cuTensorBase* eluOpGrads(cuTensorBase* X, cuTensorBase* Y, float alpha){
        float a = 1.0f, beta = 0.0f;
    
        cudnnActivationDescriptor_t eluDesc;
        cudnnCreateActivationDescriptor(&eluDesc);
        cudnnSetActivationDescriptor(eluDesc, CUDNN_ACTIVATION_ELU, CUDNN_NOT_PROPAGATE_NAN, alpha);
    
        checkCUDNN(cudnnActivationBackward(cudnnHdlG,
                                           eluDesc,
                                           &a,
                                           Y->desc.cudnnDesc,
                                           Y->data->data,
                                           Y->desc.cudnnDesc,
                                           Y->grad->data,
                                           X->desc.cudnnDesc,
                                           X->data->data,
                                           &beta,
                                           X->desc.cudnnDesc,
                                           X->grad->data))
    
        cudnnDestroyActivationDescriptor(eluDesc);
        return X;
    }
    
    cuTensorBase* swishOp(cuTensorBase* X, float beta){
        assertAllocated({X});
        assertOnSameDev({X});
        
        cudnnActivationDescriptor_t swishDesc;
        cudnnCreateActivationDescriptor(&swishDesc);
        cudnnSetActivationDescriptor(swishDesc, CUDNN_ACTIVATION_SWISH, CUDNN_NOT_PROPAGATE_NAN, beta);
        
        float alpha = 1.0f, alpha2 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                swishDesc,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data
                ))
        cudnnDestroyActivationDescriptor(swishDesc);
                
        return X;
    }
    
    cuTensorBase* swishOp(cuTensorBase* X, cuTensorBase* Y, float beta){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
        
        cudnnActivationDescriptor_t swishDesc;
        cudnnCreateActivationDescriptor(&swishDesc);
        cudnnSetActivationDescriptor(swishDesc, CUDNN_ACTIVATION_SWISH, CUDNN_NOT_PROPAGATE_NAN, beta);
        
        float alpha = 1.0f, alpha2 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                swishDesc,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha2,
                Y->desc.cudnnDesc,
                Y->data->data
                ))
                
        cudnnDestroyActivationDescriptor(swishDesc);
        return Y;
    }
    
    cuTensorBase* swishOpGrads(cuTensorBase* X, cuTensorBase* Y, float beta){
        float a = 1.0f, b = 0.0f;
    
        cudnnActivationDescriptor_t swishDesc;
        cudnnCreateActivationDescriptor(&swishDesc);
        cudnnSetActivationDescriptor(swishDesc, CUDNN_ACTIVATION_SWISH, CUDNN_NOT_PROPAGATE_NAN, beta);
    
        checkCUDNN(cudnnActivationBackward(cudnnHdlG,
                                           swishDesc,
                                           &a,
                                           Y->desc.cudnnDesc,
                                           Y->data->data,
                                           Y->desc.cudnnDesc,
                                           Y->grad->data,
                                           X->desc.cudnnDesc,
                                           X->data->data,
                                           &b,
                                           X->desc.cudnnDesc,
                                           X->grad->data))
    
        cudnnDestroyActivationDescriptor(swishDesc);
        return X;
    }
    
    cuTensorBase* clippedReluOp(cuTensorBase* X, float ceiling){
        assertAllocated({X});
        assertOnSameDev({X});
        
        cudnnActivationDescriptor_t clippedReluDesc;
        cudnnCreateActivationDescriptor(&clippedReluDesc);
        cudnnSetActivationDescriptor(clippedReluDesc, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_NOT_PROPAGATE_NAN, ceiling);
        
        float alpha = 1.0f, alpha2 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                clippedReluDesc,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha2,
                X->desc.cudnnDesc,
                X->data->data
                ))
    
        cudnnDestroyActivationDescriptor(clippedReluDesc);
        
        return X;
    }
    
    cuTensorBase* clippedReluOp(cuTensorBase* X, cuTensorBase* Y, float ceiling){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
        
        cudnnActivationDescriptor_t clippedReluDesc;
        cudnnCreateActivationDescriptor(&clippedReluDesc);
        cudnnSetActivationDescriptor(clippedReluDesc, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_NOT_PROPAGATE_NAN, ceiling);
        
        float alpha = 1.0f, alpha2 = 0.0f;
    
        checkCUDNN(cudnnActivationForward(
                cudnnHdlG,
                clippedReluDesc,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &alpha2,
                Y->desc.cudnnDesc,
                Y->data->data
                ))
                
        cudnnDestroyActivationDescriptor(clippedReluDesc);
        return Y;
    }
    
    cuTensorBase* clippedReluOpGrads(cuTensorBase* X, cuTensorBase* Y, float threshold){
        float a = 1.0f, b = 0.0f;
    
        cudnnActivationDescriptor_t clippedReluDesc;
        cudnnCreateActivationDescriptor(&clippedReluDesc);
        cudnnSetActivationDescriptor(clippedReluDesc, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_NOT_PROPAGATE_NAN, threshold);
    
        checkCUDNN(cudnnActivationBackward(cudnnHdlG,
                                           clippedReluDesc,
                                           &a,
                                           Y->desc.cudnnDesc,
                                           Y->data->data,
                                           Y->desc.cudnnDesc,
                                           Y->grad->data,
                                           X->desc.cudnnDesc,
                                           X->data->data,
                                           &b,
                                           X->desc.cudnnDesc,
                                           X->grad->data))
    
        cudnnDestroyActivationDescriptor(clippedReluDesc);
        return X;
    }
}