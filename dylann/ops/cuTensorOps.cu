//
// Created by Dylan on 8/5/2022.
//

#include <curand_kernel.h>
#include <curand_uniform.h>
#include "cuTensorOps.cuh"

namespace dylann{
    
    cuTensorBase* addOp(cuTensorBase* A, cuTensorBase* B, float alpha, float beta){
        assertAllocated({A, B});
        assertOnSameDev({A, B});
        cudaSetDevice(A->data->deviceID);
        
        //result is write back to A
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
                       &beta,
                       B->desc.cudnnDesc,
                       B->data->data,
                       &alpha,
                       A->desc.cudnnDesc,
                       A->data->data))
        return A;
    }

    cuTensorBase* addOpGrad(cuTensorBase* Y, cuTensorBase* X, float beta){
        assert(X->desc.withGrad);
        cudaSetDevice(X->data->deviceID);
    
        float a = 1.0f;
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
                                  &beta,
                                  Y->desc.cudnnDesc,
                                  Y->grad->data,
                                  &a,
                                  X->desc.cudnnDesc,
                                  X->grad->data))
                                  
        return X;
    }
    
    cuTensorBase* scale(cuTensorBase* A, float alpha){
        assertAllocated({A});
        cudaSetDevice(A->data->deviceID);
        
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                       A->desc.cudnnDesc,
                       A->data->data,
                       &alpha))
        return A;
    }
    
    cuTensorBase* scaleOpGrad(cuTensorBase* A, float alpha){
        assert(A->desc.withGrad);
        cudaSetDevice(A->data->deviceID);
    
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                    A->desc.cudnnDesc,
                                    A->grad->data,
                                    &alpha));
        
        return A;
    }
    
    //random fill with uniform distrib
    template<typename T>
    __global__ void randUniformD(T* A, unsigned int numel, long seed, double min, double max){
        unsigned int id = (threadIdx.x + blockIdx.x * blockDim.x);
        if(id >= numel) return;
        curandStateXORWOW_t state;
        curand_init(id * seed, 0, 0, &state);
        float val = curand_uniform(&state);
        A[id] = (T)(val * (max - min) + min);
    }
    
    cuTensorBase* randUniformOp(cuTensorBase* A, double min, double max){
        long seed = (long)time(nullptr);
        
        assertAllocated({A});
        cudaSetDevice(A->data->deviceID);
        
        unsigned int blockSize = 256;
        unsigned int gridSize = (A->desc.numel + blockSize - 1) / blockSize;
    
        switch (A->desc.dType) {
            case CUDNN_DATA_FLOAT :
                randUniformD<float><<<gridSize, blockSize>>>(
                (float*)A->data->data, A->desc.numel, seed, min, max);
                break;
                
            case CUDNN_DATA_DOUBLE :
                randUniformD<double><<<gridSize, blockSize>>>(
                (double*)A->data->data, A->desc.numel, seed, min, max);
                break;
                
            case CUDNN_DATA_HALF :
                randUniformD<half><<<gridSize, blockSize>>>(
                (half*)A->data->data, A->desc.numel, seed, min, max);
                break;
                
            case CUDNN_DATA_INT8 :
                randUniformD<int8_t><<<gridSize, blockSize>>>(
                (int8_t*)A->data->data, A->desc.numel, seed, min, max);
                break;
    
            case CUDNN_DATA_INT32 :
                randUniformD<int32_t><<<gridSize, blockSize>>>(
                (int32_t*)A->data->data, A->desc.numel, seed, min, max);
                break;
    
            case CUDNN_DATA_INT64 :
                randUniformD<int64_t><<<gridSize, blockSize>>>(
                (int64_t*)A->data->data, A->desc.numel, seed, min, max);
                break;
                
            default: throw std::runtime_error("unsupported dtype");
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return A;
    }
    
    //random fill with normal distrib
    template<typename T>
    __global__ void randNormalD(T* A, unsigned int numel, long seed, double mean, double stddev){
        unsigned int id = (threadIdx.x + blockIdx.x * blockDim.x);
        if(id >= numel) return;
        curandStateXORWOW_t state;
        curand_init(id * seed, 0, 0, &state);
        float val = curand_normal(&state);
        A[id] = (T)(val * stddev + mean);
    }
    
    cuTensorBase* randNormalOp(cuTensorBase* A, double mean, double stddev){
        long seed = (long)time(nullptr);
        
        assertAllocated({A});
        cudaSetDevice(A->data->deviceID);
        
        unsigned int blockSize = 256;
        unsigned int gridSize = (A->desc.numel + blockSize - 1) / blockSize;
        
        switch(A->desc.dType) {
            case CUDNN_DATA_FLOAT : randNormalD<float><<<gridSize, blockSize>>>(
                    (float *) A->data->data, A->desc.numel, seed, mean, stddev);
                    break;
                    
            case CUDNN_DATA_DOUBLE : randNormalD<double><<<gridSize, blockSize>>>(
                    (double *) A->data->data, A->desc.numel, seed, mean, stddev);
                    break;
                    
            case CUDNN_DATA_HALF : randNormalD<half><<<gridSize, blockSize>>>(
                    (half *) A->data->data, A->desc.numel, seed, mean, stddev);
                    break;
    
            case CUDNN_DATA_INT8 : randNormalD<int8_t><<<gridSize, blockSize>>>(
                    (int8_t *) A->data->data, A->desc.numel, seed, mean, stddev);
                    break;
                    
            case CUDNN_DATA_INT32 : randNormalD<int32_t><<<gridSize, blockSize>>>(
                    (int32_t *) A->data->data, A->desc.numel, seed, mean, stddev);
                    break;
                    
            case CUDNN_DATA_INT64 : randNormalD<int64_t><<<gridSize, blockSize>>>(
                    (int64_t *) A->data->data, A->desc.numel, seed, mean, stddev);
                    break;
                    
            default :  throw std::runtime_error("unsupported dtype");
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return A;
    }
    
    cuTensorBase* randNormalGradOp(cuTensorBase* A, double mean, double stddev){
        long seed = (long)time(nullptr);
        
        assertAllocated({A});
        cudaSetDevice(A->grad->deviceID);
        
        unsigned int blockSize = 256;
        unsigned int gridSize = (A->desc.numel + blockSize - 1) / blockSize;
        
        switch(A->desc.dType) {
            case CUDNN_DATA_FLOAT : randNormalD<float><<<gridSize, blockSize>>>(
                        (float *) A->grad->data, A->desc.numel, seed, mean, stddev);
                break;
            
            case CUDNN_DATA_DOUBLE : randNormalD<double><<<gridSize, blockSize>>>(
                        (double *) A->grad->data, A->desc.numel, seed, mean, stddev);
                break;
            
            case CUDNN_DATA_HALF : randNormalD<half><<<gridSize, blockSize>>>(
                        (half *) A->grad->data, A->desc.numel, seed, mean, stddev);
                break;
            
            case CUDNN_DATA_INT8 : randNormalD<int8_t><<<gridSize, blockSize>>>(
                        (int8_t *) A->grad->data, A->desc.numel, seed, mean, stddev);
                break;
            
            case CUDNN_DATA_INT32 : randNormalD<int32_t><<<gridSize, blockSize>>>(
                        (int32_t *) A->grad->data, A->desc.numel, seed, mean, stddev);
                break;
            
            case CUDNN_DATA_INT64 : randNormalD<int64_t><<<gridSize, blockSize>>>(
                        (int64_t *) A->grad->data, A->desc.numel, seed, mean, stddev);
                break;
            
            default :  throw std::runtime_error("unsupported dtype");
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return A;
    }
}