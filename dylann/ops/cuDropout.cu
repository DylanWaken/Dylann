//
// Created by Dylan on 9/7/2022.
//

#include "cuDropout.cuh"
#include <chrono>

namespace dylann {
    cuTensorBase* dropoutOp(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* mask, float b){
    
        //TODO: created device specific workspace buffer
        assertAllocated({X});
        cudaSetDevice(X->data->deviceID);
        
        cudnnDropoutDescriptor_t dropoutDesc;
        cudnnCreateDropoutDescriptor(&dropoutDesc);
        checkCUDNN(cudnnSetDropoutDescriptor(
                dropoutDesc,
                cudnnHdlG,
                b,
                nullptr,
                0,
                chrono::system_clock::now().time_since_epoch().count()
                ))
    
        checkCUDNN(cudnnDropoutForward(
                    cudnnHdlG,
                    dropoutDesc,
                    X->desc.cudnnDesc,
                    X->data->data,
                    Y->desc.cudnnDesc,
                    Y->data->data,
                    mask->data->data,
                mask->data->memSize
                ))
            
        cudnnDestroyDropoutDescriptor(dropoutDesc);
        return Y;
    }
    
    cuTensorBase* dropoutOpGrads(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* reserved, float b){
    
        cudaSetDevice(X->data->deviceID);
        
        cudnnDropoutDescriptor_t dropoutDesc;
        cudnnCreateDropoutDescriptor(&dropoutDesc);
        checkCUDNN(cudnnSetDropoutDescriptor(
                dropoutDesc,
                cudnnHdlG,
                b,
                nullptr,
                0,
                time(nullptr)
                ))
    
        checkCUDNN(cudnnDropoutBackward(
                    cudnnHdlG,
                    dropoutDesc,
                    Y->desc.cudnnDesc,
                    Y->grad->data,
                    X->desc.cudnnDesc,
                    X->grad->data,
                    cudnnWorkspaceG,
                CUDNN_WORKSPACE_SIZE_G
                ))
            
        cudnnDestroyDropoutDescriptor(dropoutDesc);
        return X;
    }
} // dylann