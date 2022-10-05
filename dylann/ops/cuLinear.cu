//
// Created by Dylan on 8/25/2022.
//

#include "cuLinear.cuh"

namespace dylann{
    /**
     * @see: https://www.adityaagrawal.net/blog/deep_learning/row_column_major
     */
    template<typename T>
    __global__ void fillBias(cuTensorBase* B, cuTensorBase* Y, float alpha2){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < B->desc.sizes.w){
            T* ptr = (T*)Y->data->data;
            #pragma unroll
            for(auto i = 0; i < Y->desc.sizes.n; i++){
                ptr[i * Y->desc.sizes.w + idx] = ((T*)B->data->data)[idx];
            }
        }
    }
    
    template<typename T>
    __global__ void calcBiasGrad(cuTensorBase* B, cuTensorBase* Y){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < B->desc.sizes.w){
            T* ptr = (T*)Y->grad->data;
            T sum = 0;
            #pragma unroll
            for(auto i = 0; i < Y->desc.sizes.n; i++){
                sum += ptr[i * Y->desc.sizes.w + idx];
            }
            ((T*)B->grad->data)[idx] = sum;
        }
    }
    
    void FLOAT_LINEAR(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2){
        //forward gemm (linear operation)
        //float a = 1.0f, b = 1.0f;
        
        checkCUBLAS(cublasSgemm_v2(cublasHdlG,
                                   CUBLAS_OP_T,   //row major to column major for weights
                                   CUBLAS_OP_N,   //read the original row major as column major, auto trans
                                   Y->desc.sizes.w,
                                   X->desc.sizes.n,
                                   X->desc.sizes.w,
                                   &alpha1,
                                   (float*)W->data->data,
                                   W->desc.sizes.w,
                                   (float*)X->data->data,
                                   X->desc.sizes.w,
                                   &alpha2,
                                   (float*)Y->data->data,
                                   Y->desc.sizes.w
        ))
    }
    
    void FLOAT_LINEAR_GRAD_X(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2){
    
        checkCUBLAS(cublasSgemm_v2(cublasHdlG,
                                   CUBLAS_OP_N,  //auto transpose for weights
                                   CUBLAS_OP_N,
                                   W->desc.sizes.w,
                                   Y->desc.sizes.n,
                                   W->desc.sizes.h,
                                   &alpha1,
                                   (float*)W->data->data,
                                   W->desc.sizes.w,
                                   (float*)Y->grad->data,
                                   Y->desc.sizes.w,
                                   &alpha2,
                                   (float*)X->grad->data,
                                   X->desc.sizes.w
        ))
    }
    
    void FLOAT_LINEAR_GRAD_W(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2){
    
        checkCUBLAS(cublasSgemm_v2(cublasHdlG,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_T,  //we need to recover the "row major"
                                   X->desc.sizes.w,
                                   Y->desc.sizes.w,
                                   X->desc.sizes.n,
                                   &alpha1,
                                   (float*)X->data->data,
                                   X->desc.sizes.w,
                                   (float*)Y->grad->data,
                                   Y->desc.sizes.w,
                                  &alpha2,
                                  (float*)W->grad->data,
                                  W->desc.sizes.w
                                   ))
    }
    
    void HALF_LINEAR(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2){
        //forward gemm (linear operation)
        half a = __float2half(alpha1);
        half b = __float2half(alpha2);
    
        checkCUBLAS(cublasHgemm(
                cublasHdlG,
                CUBLAS_OP_T,   //row major to column major for weights
                CUBLAS_OP_N,   //read the original row major as column major, auto
                Y->desc.sizes.w,
                X->desc.sizes.n,
                X->desc.sizes.w,
                &a,
                (half*)W->data->data,
                W->desc.sizes.w,
                (half*)X->data->data,
                X->desc.sizes.w,
                &b,
                (half*)Y->data->data,
                Y->desc.sizes.w
                ))
    }
    
    void HALF_LINEAR_GRAD_X(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2){
        half a = __float2half(alpha1), b = __float2half(alpha2);
    
        checkCUBLAS(cublasHgemm(cublasHdlG,
                                CUBLAS_OP_N,  //auto transpose for weights
                                CUBLAS_OP_N,
                                W->desc.sizes.w,
                                Y->desc.sizes.n,
                                W->desc.sizes.h,
                                &a,
                                (half*)W->data->data,
                                W->desc.sizes.w,
                                (half*)Y->grad->data,
                                Y->desc.sizes.w,
                                &b,
                                (half*)X->grad->data,
                                X->desc.sizes.w
        ))
    }
    
    void HALF_LINEAR_GRAD_W(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2){
        half a = __float2half(alpha1), b = __float2half(alpha2);
        
        checkCUBLAS(cublasHgemm(cublasHdlG,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,  //we need to recover the "row major"
                                X->desc.sizes.w,
                                Y->desc.sizes.w,
                                X->desc.sizes.n,
                                &a,
                                (half*)X->data->data,
                                X->desc.sizes.w,
                                (half*)Y->grad->data,
                                Y->desc.sizes.w,
                                &b,
                                (half*)W->grad->data,
                                W->desc.sizes.w
        ))
    }
    
    void DOUBLE_LINEAR(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2){
        //forward gemm (linear operation)
        auto a = (double)alpha1, b = (double )alpha2;
    
        checkCUBLAS(cublasDgemm_v2(cublasHdlG,
                                   CUBLAS_OP_T,   //row major to column major for weights
                                   CUBLAS_OP_N,   //read the original row major as column major, auto trans
                                   Y->desc.sizes.w,
                                   X->desc.sizes.n,
                                   X->desc.sizes.w,
                                   &a,
                                   (double*)W->data->data,
                                   W->desc.sizes.w,
                                   (double*)X->data->data,
                                   X->desc.sizes.w,
                                   &b,
                                   (double*)Y->data->data,
                                   Y->desc.sizes.w
                                   ))
    }
    
    void DOUBLE_LINEAR_GRAD_X(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2){
        auto a = (double)alpha1, b = (double)alpha2;
        
        checkCUBLAS(cublasDgemm_v2(cublasHdlG,
                                   CUBLAS_OP_N,  //auto transpose for weights
                                   CUBLAS_OP_N,
                                   W->desc.sizes.w,
                                   Y->desc.sizes.n,
                                   W->desc.sizes.h,
                                   &a,
                                   (double*)W->data->data,
                                   W->desc.sizes.w,
                                   (double*)Y->grad->data,
                                   Y->desc.sizes.w,
                                   &b,
                                   (double*)X->grad->data,
                                   X->desc.sizes.w
        ))
    }
    
    void DOUBLE_LINEAR_GRAD_W(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2){
        auto a = (double)alpha1, b = (double)alpha2;
    
        checkCUBLAS(cublasDgemm_v2(cublasHdlG,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_T,  //we need to recover the "row major"
                                   X->desc.sizes.w,
                                   Y->desc.sizes.w,
                                   X->desc.sizes.n,
                                   &a,
                                   (double*)X->data->data,
                                   X->desc.sizes.w,
                                   (double*)Y->grad->data,
                                   Y->desc.sizes.w,
                                   &b,
                                   (double*)W->grad->data,
                                   W->desc.sizes.w
        ))
    }
    
    cuTensorBase *linearOp(cuTensorBase* W, cuTensorBase* B, cuTensorBase* X, cuTensorBase* Y, float alpha1, float alpha2){
    
        cudaSetDevice(W->data->deviceID);
        
        //set cublas
        checkCUBLAS(cublasSetMathMode(cublasHdlG, CUBLAS_TENSOR_OP_MATH))
    
        unsigned int block = 256;
        unsigned int grid = (B->desc.sizes.w + block - 1) / block;

        switch (Y->desc.dType) {
            case CUDNN_DATA_FLOAT:
                fillBias<float><<<grid, block>>>(B, Y, alpha2);
                break;
            case CUDNN_DATA_HALF:
                fillBias<half><<<grid, block>>>(B, Y, alpha2);
                break;
            case CUDNN_DATA_DOUBLE:
                fillBias<double><<<grid, block>>>(B, Y, alpha2);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }

        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        
        //assert same dtye
        assert(W->desc.dType == X->desc.dType
           && W->desc.dType == Y->desc.dType);
        
        //forward gemm (linear operation)
        switch (X->desc.dType) {
            case CUDNN_DATA_FLOAT:
                FLOAT_LINEAR(W, X, Y, alpha1, 1);
                break;
            case CUDNN_DATA_HALF:
                HALF_LINEAR(W, X, Y, alpha1, 1);
                break;
            case CUDNN_DATA_DOUBLE:
                DOUBLE_LINEAR(W, X, Y, alpha1, 1);
                break;
            default:
                throw std::runtime_error("unsupported dtype");
        }
        
        return Y;
    }
    cuTensorBase *linearOpGrads(cuTensorBase* W, cuTensorBase* B, cuTensorBase* X, cuTensorBase* Y,
                                float alpha1, float alpha2){
        checkCUBLAS(cublasSetMathMode(cublasHdlG, CUBLAS_TENSOR_OP_MATH))
    
        //assert same dtype
        assert(Y->desc.dType == X->desc.dType
               && Y->desc.dType == Y->desc.dType);
    
        cudaSetDevice(W->data->deviceID);
    
        //forward gradient for features
        switch (X->desc.dType) {
            case CUDNN_DATA_FLOAT:
                FLOAT_LINEAR_GRAD_X(W, X, Y, alpha1, alpha2);
                break;
            case CUDNN_DATA_HALF:
                HALF_LINEAR_GRAD_X(W, X, Y, alpha1, alpha2);
                break;
            case CUDNN_DATA_DOUBLE:
                DOUBLE_LINEAR_GRAD_X(W, X, Y, alpha1, alpha2);
                break;
            default:
                throw std::runtime_error("unsupported dtype");
        }
    
        //forward gradients for weights
        switch (X->desc.dType) {
            case CUDNN_DATA_FLOAT:
                FLOAT_LINEAR_GRAD_W(W, X, Y, alpha1, alpha2);
                break;
            case CUDNN_DATA_HALF:
                HALF_LINEAR_GRAD_W(W, X, Y, alpha1, alpha2);
                break;
            case CUDNN_DATA_DOUBLE:
                DOUBLE_LINEAR_GRAD_W(W, X, Y, alpha1, alpha2);
                break;
            default:
                throw std::runtime_error("unsupported dtype");
        }
    
        unsigned int block = 256;
        unsigned int grid = (B->desc.sizes.w + block - 1) / block;
        
        //forward gradients for biases
        switch (X->desc.dType) {
            case CUDNN_DATA_FLOAT:
                calcBiasGrad<float><<<grid, block>>>(B,Y);
                break;
            case CUDNN_DATA_HALF:
                calcBiasGrad<__half><<<grid, block>>>(B,Y);
                break;
            case CUDNN_DATA_DOUBLE:
                calcBiasGrad<double><<<grid, block>>>(B,Y);
                break;
            default:
                throw std::runtime_error("unsupported dtype");
        }
        
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        
        return Y;
    }
}