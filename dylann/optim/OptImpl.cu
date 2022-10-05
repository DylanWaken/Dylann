//
// Created by Dylan on 10/4/2022.
//

#include "OptImpl.cuh"

namespace dylann {
    template<typename T>
    __global__ void momentumApplyD(cuTensorBase* X, cuTensorBase* m,float LEARNING_RATE, float BETA,
                                   float L2, bool isWeight) {
        unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < X->desc.sizes.size){
            
            auto xPtr = (T*)X->data->data;
            auto mPtr = (T*)m->data->data;
            auto xGradPtr = (T*)X->grad->data;
            
            float mVal = BETA * mPtr[idx] + (1-BETA) * xGradPtr[idx];
            xPtr[idx] = isWeight? xPtr[idx] * (1-L2) - LEARNING_RATE * mVal
                    : xPtr[idx] - LEARNING_RATE * mVal;
            mPtr[idx] = mVal;
        }
    }
    
    __global__ void halfMomentumApplyD(cuTensorBase* X, cuTensorBase* m, float LEARNING_RATE, float BETA,
                                       float L2, bool isWeight) {
        unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < X->desc.sizes.size){
            
            auto xPtr = (half*)X->data->data;
            auto mPtr = (half*)m->data->data;
            auto xGradPtr = (half*)X->grad->data;
            
            half lr = __float2half(LEARNING_RATE);
            half beta = __float2half(BETA);
            half l2 = __float2half(L2);
            half one = __float2half(1);
            
            half mVal = __hmul(beta, mPtr[idx]) + __hmul((one-beta), xGradPtr[idx]);
            xPtr[idx] = isWeight? __hmul(xPtr[idx], (one-l2)) - __hmul(lr, mVal)
                    : xPtr[idx] - __hmul(lr, mVal);
            mPtr[idx] = mVal;
        }
    }
    
    void momentumApply(cuTensorBase* X, cuTensorBase* m, float LEARNING_RATE, float BETA,
                       float L2, bool isWeight) {
        unsigned int blockSize = 256;
        unsigned int gridSize = (X->desc.sizes.size + blockSize - 1) / blockSize;
        
        switch (X->desc.dType) {
            case CUDNN_DATA_FLOAT:
                momentumApplyD<float><<<gridSize, blockSize>>>(X, m, LEARNING_RATE, BETA, L2, isWeight);
                break;
            case CUDNN_DATA_DOUBLE:
                momentumApplyD<double><<<gridSize, blockSize>>>(X, m, LEARNING_RATE, BETA, L2, isWeight);
                break;
            case CUDNN_DATA_HALF:
                halfMomentumApplyD<<<gridSize, blockSize>>>(X, m, LEARNING_RATE, BETA, L2, isWeight);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
} // dylann