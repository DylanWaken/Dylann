//
// Created by Dylan on 8/5/2022.
//

#include <curand_kernel.h>
#include <curand_uniform.h>
#include "cuTensorOps.cuh"

namespace dylann{
    
    __device__ double sqrtD(float x){
        return pow(x, 0.5);
    }
    
    __device__ double sqrtD(double x){
        return pow(x, 0.5);
    }
    
    __device__ double sqrtD(half x){
        return pow(__half2float(x), 0.5);
    }
    
    __device__ double sqrtD(int8_t x){
        return pow(x, 0.5);
    }
    
    __device__ double sqrtD(int32_t x){
        return pow(x, 0.5);
    }
    
    __device__ double sqrtD(int64_t x){
        return pow(x, 0.5);
    }
    
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

    cuTensorBase* addOpGrad(cuTensorBase* A, cuTensorBase* B, float alpha, float beta){
        assert(A->desc.withGrad);
        cudaSetDevice(A->data->deviceID);
        
        float a = 1.0f;
        //add grads in A to B
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
                                  &beta,
                                  A->desc.cudnnDesc,
                                  A->grad->data,
                                  &a,
                                  B->desc.cudnnDesc,
                                  B->grad->data))
    
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                    A->desc.cudnnDesc,
                                    A->data->data,
                                    &alpha))
                                    
        return A;
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
    
    cuTensorBase* flattenOp(cuTensorBase* X, cuTensorBase* Y){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
        cudaSetDevice(X->data->deviceID);
    
        assert(X->desc.sizes.n == Y->desc.sizes.n);
        assert(X->desc.sizes.size == Y->desc.sizes.size);
    
        cudaMemcpy(Y->data->data, X->data->data, X->data->memSize, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        return Y;
    };
    
    cuTensorBase* flattenOpGrad(cuTensorBase* X, cuTensorBase* Y){
        cudaSetDevice(X->data->deviceID);
        
        cudaMemcpy(X->grad->data, Y->grad->data, X->grad->memSize, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        return Y;
    };
    
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
            
            default :  throw std::runtime_error("unsupported dtype");
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return A;
    }
    
    template<typename T>
    __global__ void hadamardD(TDescriptor &desc, T* X, T* Y){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < desc.numel){
            Y[idx] = X[idx] * Y[idx];
        }
    }
    
    void hadamard(cudnnDataType_t dtype, void* X, void* Y, unsigned int size){
        unsigned int blockSize = 256;
        unsigned int gridSize = (size + blockSize - 1) / blockSize;
        switch(dtype){
            case CUDNN_DATA_FLOAT : hadamardD<float><<<gridSize, blockSize>>>(
                    *(TDescriptor*)X, (float*)X, (float*)Y);
                break;
            case CUDNN_DATA_DOUBLE : hadamardD<double><<<gridSize, blockSize>>>(
                    *(TDescriptor*)X, (double*)X, (double*)Y);
                break;
            case CUDNN_DATA_HALF : hadamardD<half><<<gridSize, blockSize>>>(
                    *(TDescriptor*)X, (half*)X, (half*)Y);
                break;
            default : throw std::runtime_error("unsupported dtype");
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
    
    cuTensorBase* hadamardOp(cuTensorBase* A, cuTensorBase* B){
        assertAllocated({A, B});
        assertOnSameDev({A, B});
        cudaSetDevice(A->data->deviceID);
        hadamard(A->desc.dType, A->data->data, B->data->data, A->desc.numel);
        return A;
    }
    
    template<typename T>
    __global__ void RSMPropVD( T* V, T* G, float BETA, unsigned int size){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < size){
            V[idx] = (T)BETA * V[idx] + (T)(1 - BETA) * G[idx] * G[idx];
        }
    }
    
    template<typename T>
    __global__ void RSMPropAD(T* A, T* V, T* G, float LR, unsigned int size, float L2, float EPSILON){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < size){
            A[idx] = A[idx] * (T)(1 - L2) - (T)LR * G[idx] / ((T)sqrtD(V[idx]) + (T)(EPSILON));
        }
    }
    
    void RSMPropV(cudnnDataType_t dtype, void* V, void* G, float BETA, unsigned int size){
        unsigned int blockSize = 256;
        unsigned int gridSize = (size + blockSize - 1) / blockSize;
        switch(dtype){
            case CUDNN_DATA_FLOAT : RSMPropVD<float><<<gridSize, blockSize>>>(
                     (float*)V, (float*)G, BETA, size);
                break;
            case CUDNN_DATA_DOUBLE : RSMPropVD<double><<<gridSize, blockSize>>>(
                     (double*)V, (double*)G, BETA, size);
                break;
            case CUDNN_DATA_HALF : RSMPropVD<half><<<gridSize, blockSize>>>(
                     (half*)V, (half*)G, BETA, size);
                break;
            default : throw std::runtime_error("unsupported dtype");
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
    
    void RSMPropA(cudnnDataType_t dtype, void* W, void* V, void* G, float EPSILON, float LR, float L2, unsigned int size){
        unsigned int blockSize = 256;
        unsigned int gridSize = (size + blockSize - 1) / blockSize;
        switch(dtype){
            case CUDNN_DATA_FLOAT : RSMPropAD<float><<<gridSize, blockSize>>>(
                    (float*)W, (float*)V, (float*)G, LR, size, L2, EPSILON);
                break;
            case CUDNN_DATA_DOUBLE : RSMPropAD<double><<<gridSize, blockSize>>>(
                     (double*)W, (double*)V, (double*)G, LR, size, L2, EPSILON);
                break;
            case CUDNN_DATA_HALF : RSMPropAD<half><<<gridSize, blockSize>>>(
                     (half*)W, (half*)V, (half*)G, LR, size, L2, EPSILON);
                break;
            default : throw std::runtime_error("unsupported dtype");
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
    
    template<typename T>
    __global__ void AdamAD(cudnnDataType_t dtype, T* A, T* M, T* V, float LR,  float L2, unsigned int size, float EPSILON){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < size){
            A[idx] = A[idx] - (T)LR * M[idx] / ((T)sqrtD(V[idx]) + (T)(EPSILON));
        }
    }
    
    
    void AdamA(cudnnDataType_t dtype, void* W, void* M, void* V, float EPSILON, float LR, float L2, unsigned int size){
        unsigned int blockSize = 256;
        unsigned int gridSize = (size + blockSize - 1) / blockSize;
        switch(dtype){
            case CUDNN_DATA_FLOAT : AdamAD<float><<<gridSize, blockSize>>>(
                    dtype, (float*)W, (float*)M, (float*)V, LR, L2, size, EPSILON);
                break;
            case CUDNN_DATA_DOUBLE : AdamAD<double><<<gridSize, blockSize>>>(
                    dtype, (double*)W, (double*)M, (double*)V, LR, L2, size, EPSILON);
                break;
            case CUDNN_DATA_HALF : AdamAD<half><<<gridSize, blockSize>>>(
                    dtype, (half*)W, (half*)M, (half*)V, LR, L2, size, EPSILON);
                break;
            default : throw std::runtime_error("unsupported dtype");
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
    
    cuTensorBase* initParamOp(cuTensorBase* A) {
        switch (A->desc.paramInitType) {
            case INIT_ZERO:
                randNormalOp(A, 0, 0);
                break;
            case INIT_XAVIER_LINEAR_WEIGHT:
            case INIT_XAVIER_LINEAR_BIAS:
                //Xavier Initialization : W ~ U(-1/sqrt(n), 1/sqrt(n))
                randUniformOp(A, -1.0 / sqrt(A->desc.sizes.w), 1.0 / sqrt(A->desc.sizes.w));
                break;
        
            case INIT_STD_CONV_WEIGHT:
                //torch default : W ~ N(0, 2/sqrt(fh*fw*in))
                randNormalOp(A, 0, 2.0 / sqrt(A->desc.sizes.h * A->desc.sizes.w * A->desc.sizes.c));
                break;
        
            case INIT_STD_CONV_BIAS:
                //torch default : B ~ N(0, 2/sqrt(in))
                randNormalOp(A, 0, 2.0 / sqrt(A->desc.sizes.c));
                break;
        
            case INIT_STD_BN_WEIGHT:
                //set 1
                randNormalOp(A, 1, 0);
                break;
        
            case INIT_STD_BN_BIAS:
                //set 0
                randNormalOp(A, 0, 0);
                break;
        }
        return A;
    }
}