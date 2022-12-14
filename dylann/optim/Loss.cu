//
// Created by Dylan on 9/17/2022.
//

#include "Loss.cuh"

namespace dylann {
    
    template<typename T>
    __global__ void crossEntropyD(T* pred, T* target, unsigned int size) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx<size) {
             pred[idx] = -target[idx] * (T)log(pred[idx]);
        }
    }
    
    template<typename T>
    __global__ void mseD(T* pred, T* target, unsigned int size) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx<size) {
            pred[idx] = 0.5 * pow( pred[idx] - target[idx], 2);
        }
    }
    
    float MSE::loss(cuTensorBase *target) {
        float lossHost = 0;
        double lossHostD = 0;
        half lossHostH = 0;
        
        if(calcBuf == nullptr){
            calcBuf = cuTensor::create(pred->data->deviceID, pred->desc.dType, pred->desc.sizes).impl;
            lossVal = cuTensor::create(pred->data->deviceID, pred->desc.dType, 1).impl;
        }
    
        cudaMemcpy(calcBuf->data->data, pred->data->data, pred->data->memSize, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        
        unsigned int blockSize = 256;
        unsigned int gridSize = (pred->desc.sizes.size + blockSize - 1) / blockSize;
        
        switch (pred->desc.dType) {
            case CUDNN_DATA_FLOAT:
                mseD<float><<<gridSize, blockSize>>>((float*)calcBuf->data->data, (float*)target->data->data, pred->desc.sizes.size);
                break;
            case CUDNN_DATA_DOUBLE:
                mseD<double><<<gridSize, blockSize>>>((double*)calcBuf->data->data, (double*)target->data->data, pred->desc.sizes.size);
                break;
            case CUDNN_DATA_HALF:
                mseD<half><<<gridSize, blockSize>>>((half*)calcBuf->data->data, (half*)target->data->data, pred->desc.sizes.size);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    
        reduceOp(calcBuf, lossVal, (int)calcBuf->desc.sizes.size);
        assertCuda(__FILE__, __LINE__);
    
        switch (pred->desc.dType) {
            case CUDNN_DATA_FLOAT:
                cudaMemcpy(&lossHost, lossVal->data->data, calcBuf->desc.elementSize, cudaMemcpyDeviceToHost);
                assertCuda(__FILE__, __LINE__);
                return lossHost;
            case CUDNN_DATA_DOUBLE:
                cudaMemcpy(&lossHostD, lossVal->data->data, calcBuf->desc.elementSize, cudaMemcpyDeviceToHost);
                assertCuda(__FILE__, __LINE__);
                return (float)lossHostD;
            case CUDNN_DATA_HALF:
                cudaMemcpy(&lossHostH, lossVal->data->data, calcBuf->desc.elementSize, cudaMemcpyDeviceToHost);
                assertCuda(__FILE__, __LINE__);
                return (float)lossHostH;
            default:
                throw std::runtime_error("Unsupported data type");
        }
    }
    
    cuTensorBase *MSE::backward(cuTensorBase *target) {
        cudaMemcpy(pred->grad->data, pred->data->data, pred->data->memSize, cudaMemcpyDeviceToDevice);
    
        float a = 1.0f, a2 = -1.0f;
        checkCUDNN(cudnnAddTensor(
                cudnnHdlG,
                &a2,
                pred->desc.cudnnDesc,
                target->data->data,
                &a,
                pred->desc.cudnnDesc,
                pred->grad->data
                ))
                
        return pred;
    }
    
    float CrossEntropy::loss(cuTensorBase *target) {
        if(calcBuf == nullptr){
            calcBuf = cuTensor::create(pred->data->deviceID, pred->desc.dType, pred->desc.sizes).impl;
            lossVal = cuTensor::create(pred->data->deviceID, pred->desc.dType, 1).impl;
            cudaMallocHost(&lossHost, pred->desc.elementSize);
            assertCuda(__FILE__, __LINE__);
        }
    
        cudaMemcpy(calcBuf->data->data, pred->data->data, pred->data->memSize, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
    
        unsigned int blockSize = 256;
        unsigned int gridSize = (pred->data->memSize + blockSize - 1) / blockSize;
    
        switch (pred->desc.dType) {
            case CUDNN_DATA_FLOAT:
                crossEntropyD<float><<<gridSize, blockSize>>>((float*)calcBuf->data->data,
                                  (float*)target->data->data, pred->data->memSize/sizeof(float));
                break;
            case CUDNN_DATA_DOUBLE:
                crossEntropyD<double><<<gridSize, blockSize>>>((double*)calcBuf->data->data,
                                        (double*)target->data->data, pred->data->memSize/sizeof(double));
                break;
            case CUDNN_DATA_HALF:
                crossEntropyD<half><<<gridSize, blockSize>>>((half*)calcBuf->data->data,
                                       (half*)target->data->data, pred->data->memSize/sizeof(half));
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    
        reduceOp(calcBuf, lossVal, (int)calcBuf->desc.sizes.size);
        assertCuda(__FILE__, __LINE__);
        
        cudaMemcpy(lossHost, lossVal->data->data, calcBuf->desc.elementSize, cudaMemcpyDeviceToHost);
        assertCuda(__FILE__, __LINE__);
    
        switch (pred->desc.dType) {
            case CUDNN_DATA_FLOAT:
                return *((float*)lossHost)/(float)pred->desc.sizes.n;
            case CUDNN_DATA_DOUBLE:
                return (float)*((double*)lossHost)/(float)pred->desc.sizes.n;
            case CUDNN_DATA_HALF:
                return (float)*((half*)lossHost)/(float)pred->desc.sizes.n;
            default:
                throw std::runtime_error("Unsupported data type");
        }
    }
    
    //The actual subtract calc will be at the softmaxCE operand
    cuTensorBase *CrossEntropy::backward(cuTensorBase *target) {
        float af = 1.0f, bf = -1.0f;
    
        cudaMemcpy(pred->grad->data, pred->data->data, pred->data->memSize, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
    
        checkCUDNN(cudnnAddTensor(
                cudnnHdlG,
                &bf,
                target->desc.cudnnDesc,
                target->data->data,
                &af,
                pred->desc.cudnnDesc,
                pred->grad->data
        ))
        
        return pred;
    }
} // dylann