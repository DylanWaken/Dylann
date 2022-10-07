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
    
        size_t stateSize;
        cudnnDropoutGetStatesSize(cudnnHdlG, &stateSize);
        
        checkCUDNN(cudnnSetDropoutDescriptor(
                dropoutDesc,
                cudnnHdlG,
                b,
                mask->data->data,
                stateSize,
                chrono::system_clock::now().time_since_epoch().count()
                ))
    
        checkCUDNN(cudnnDropoutForward(
                    cudnnHdlG,
                    dropoutDesc,
                    X->desc.cudnnDesc,
                    X->data->data,
                    Y->desc.cudnnDesc,
                    Y->data->data,
                   (char*)mask->data->data + stateSize,
                   mask->data->memSize - stateSize
                ))
            
        cudnnDestroyDropoutDescriptor(dropoutDesc);
        return Y;
    }
    
    cuTensorBase* dropoutOpGrads(cuTensorBase* X, cuTensorBase* Y, cuTensorBase* mask, float b){
    
        cudaSetDevice(X->data->deviceID);
        
        cudnnDropoutDescriptor_t dropoutDesc;
        cudnnCreateDropoutDescriptor(&dropoutDesc);
    
        size_t stateSize;
        cudnnDropoutGetStatesSize(cudnnHdlG, &stateSize);
        
        checkCUDNN(cudnnSetDropoutDescriptor(
                dropoutDesc,
                cudnnHdlG,
                b,
                mask->data->data,
                stateSize,
                chrono::system_clock::now().time_since_epoch().count()
                ))
    
        checkCUDNN(cudnnDropoutBackward(
                    cudnnHdlG,
                    dropoutDesc,
                    Y->desc.cudnnDesc,
                    Y->grad->data,
                    X->desc.cudnnDesc,
                    X->grad->data,
                    (char*)mask->data->data + stateSize,
                    mask->data->memSize - stateSize
                ))
            
        cudnnDestroyDropoutDescriptor(dropoutDesc);
        return X;
    }
} // dylann