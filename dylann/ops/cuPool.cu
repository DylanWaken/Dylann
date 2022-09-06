//
// Created by Dylan on 9/4/2022.
//

#include "cuPool.cuh"

namespace dylann {
    cuTensorBase *maxPoolOp(cuTensorBase* X, cuTensorBase* Y, int rangeH, int rangeW,
                            int padH, int padW, int strideH, int strideW){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
        
        cudaSetDevice(X->data->deviceID);
        
        cudnnPoolingDescriptor_t poolDesc;
        cudnnCreatePoolingDescriptor(&poolDesc);
        cudnnSetPooling2dDescriptor(poolDesc,
                                    CUDNN_POOLING_MAX,
                                    CUDNN_NOT_PROPAGATE_NAN,
                                    rangeH,
                                    rangeW,
                                    padH,
                                    padW,
                                    strideH,
                                    strideW);
    
        float alpha = 1.0f, beta = 0.0f;
        
        checkCUDNN(cudnnPoolingForward(
                cudnnHdlG,
                poolDesc,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &beta,
                Y->desc.cudnnDesc,
                Y->data->data
                ))
    
        cudnnDestroyPoolingDescriptor(poolDesc);
        
        return Y;
    }
    
    cuTensorBase *maxPoolOpGrads(cuTensorBase* X, cuTensorBase* Y,  int rangeH, int rangeW,
                                 int padH, int padW, int strideH, int strideW){
        cudaSetDevice(X->data->deviceID);
    
        cudnnPoolingDescriptor_t poolDesc;
        cudnnCreatePoolingDescriptor(&poolDesc);
        cudnnSetPooling2dDescriptor(poolDesc,
                                    CUDNN_POOLING_MAX,
                                    CUDNN_NOT_PROPAGATE_NAN,
                                    rangeH,
                                    rangeW,
                                    padH,
                                    padW,
                                    strideH,
                                    strideW);
        
        float alpha = 1.0f, beta = 1.0f;
        
        checkCUDNN(cudnnPoolingBackward(
                cudnnHdlG,
                poolDesc,
                &alpha,
                Y->desc.cudnnDesc,
                Y->data->data,
                Y->desc.cudnnDesc,
                Y->grad->data,
                X->desc.cudnnDesc,
                X->data->data,
                &beta,
                X->desc.cudnnDesc,
                X->grad->data
                ))
        
        cudnnDestroyPoolingDescriptor(poolDesc);
        
        return X;
    }
    
    cuTensorBase *avgPoolOp(cuTensorBase* X, cuTensorBase* Y, int rangeH, int rangeW,
                            int padH, int padW, int strideH, int strideW){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
    
        cudaSetDevice(X->data->deviceID);
    
        cudnnPoolingDescriptor_t poolDesc;
        cudnnCreatePoolingDescriptor(&poolDesc);
        cudnnSetPooling2dDescriptor(poolDesc,
                                    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                    CUDNN_NOT_PROPAGATE_NAN,
                                    rangeH,
                                    rangeW,
                                    padH,
                                    padW,
                                    strideH,
                                    strideW);
    
        float alpha = 1.0f, beta = 0.0f;
    
        checkCUDNN(cudnnPoolingForward(
                cudnnHdlG,
                poolDesc,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &beta,
                Y->desc.cudnnDesc,
                Y->data->data
        ))
    
        cudnnDestroyPoolingDescriptor(poolDesc);
    
        return Y;
    }
    
    cuTensorBase *avgPoolOpGrads(cuTensorBase* X, cuTensorBase* Y,  int rangeH, int rangeW,
                                 int padH, int padW, int strideH, int strideW){
        cudaSetDevice(X->data->deviceID);
    
        cudnnPoolingDescriptor_t poolDesc;
        cudnnCreatePoolingDescriptor(&poolDesc);
        cudnnSetPooling2dDescriptor(poolDesc,
                                    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                    CUDNN_NOT_PROPAGATE_NAN,
                                    rangeH,
                                    rangeW,
                                    padH,
                                    padW,
                                    strideH,
                                    strideW);
    
        float alpha = 1.0f, beta = 1.0f;
    
        checkCUDNN(cudnnPoolingBackward(
                cudnnHdlG,
                poolDesc,
                &alpha,
                Y->desc.cudnnDesc,
                Y->data->data,
                Y->desc.cudnnDesc,
                Y->grad->data,
                X->desc.cudnnDesc,
                X->data->data,
                &beta,
                X->desc.cudnnDesc,
                X->grad->data
        ))
    
        cudnnDestroyPoolingDescriptor(poolDesc);
        
        return X;
    }
    
    cuTensorBase *globalAvgPoolOp(cuTensorBase* X, cuTensorBase* Y){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
    
        cudaSetDevice(X->data->deviceID);
    
        cudnnPoolingDescriptor_t poolDesc;
        cudnnCreatePoolingDescriptor(&poolDesc);
        cudnnSetPooling2dDescriptor(poolDesc,
                                    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                    CUDNN_NOT_PROPAGATE_NAN,
                                    (int)X->desc.sizes.h,
                                    (int)X->desc.sizes.w,
                                    0,
                                    0,
                                    1,
                                    1);
    
        float alpha = 1.0f, beta = 0.0f;
    
        checkCUDNN(cudnnPoolingForward(
                cudnnHdlG,
                poolDesc,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                &beta,
                Y->desc.cudnnDesc,
                Y->data->data
        ))
    
        cudnnDestroyPoolingDescriptor(poolDesc);
    
        return Y;
    }
    
    cuTensorBase *globalAvgPoolOpGrads(cuTensorBase* X, cuTensorBase* Y){
        cudaSetDevice(X->data->deviceID);
    
        cudnnPoolingDescriptor_t poolDesc;
        cudnnCreatePoolingDescriptor(&poolDesc);
        cudnnSetPooling2dDescriptor(poolDesc,
                                    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                    CUDNN_NOT_PROPAGATE_NAN,
                                    (int)X->desc.sizes.h,
                                    (int)X->desc.sizes.w,
                                    0,
                                    0,
                                    1,
                                    1);
    
        float alpha = 1.0f, beta = 1.0f;
    
        checkCUDNN(cudnnPoolingBackward(
                cudnnHdlG,
                poolDesc,
                &alpha,
                Y->desc.cudnnDesc,
                Y->data->data,
                Y->desc.cudnnDesc,
                Y->grad->data,
                X->desc.cudnnDesc,
                X->data->data,
                &beta,
                X->desc.cudnnDesc,
                X->grad->data
        ))
    
        cudnnDestroyPoolingDescriptor(poolDesc);
        
        return X;
    }
} // dylann