//
// Created by Dylan on 8/25/2022.
//

#include "cuLinear.cuh"

namespace dylann{
    
    void fillBias(cuTensorBase* B, cuTensorBase* Y){
        for(auto i = 0; i < Y->desc.sizes.h; i++){
            auto offset = i * Y->desc.sizes.w;
            char* destPtr = (char*)Y->data->data + offset * Y->desc.elementSize;
            cudaMemcpy(destPtr, B->data->data, B->data->memSize, cudaMemcpyDeviceToDevice);
        }
        assertCuda(__FILE__, __LINE__);
    };
    
    void FLOAT_LINEAR(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y){
        //run gemm (linear operation)
        float a = 1.0f, b = 1.0f;
    
        checkCUBLAS(cublasSgemm_v2(cublasHdlG,
                                   CUBLAS_OP_T,   //row major to column major for weights
                                   CUBLAS_OP_N,   //read the original row major as column major, auto trans
                                   Y->desc.sizes.w,
                                   X->desc.sizes.h,
                                   X->desc.sizes.w,
                                   &a,
                                   (float*)W->data->data,
                                   W->desc.sizes.h,
                                   (float*)X->data->data,
                                   X->desc.sizes.w,
                                   &b,
                                   (float*)Y->data->data,
                                   Y->desc.sizes.w
        ))
    }
    
    void FLOAT_LINEAR_GRAD_X(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y){
        float a = 1.0f, b = 1.0f;
    
        checkCUBLAS(cublasSgemm_v2(cublasHdlG,
                                   CUBLAS_OP_N,  //auto transpose for weights
                                   CUBLAS_OP_N,
                                   W->desc.sizes.h,
                                   Y->desc.sizes.h,
                                   W->desc.sizes.w,
                                   &a,
                                   (float*)W->data->data,
                                   W->desc.sizes.h,
                                   (float*)Y->grad->data,
                                   Y->desc.sizes.w,
                                   &b,
                                   (float*)X->grad->data,
                                   X->desc.sizes.w
        ))
    }
    
    void FLOAT_LINEAR_GRAD_W(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y){
        float a = 1.0f, b = 0.0f;
    
        checkCUBLAS(cublasSgemm_v2(cublasHdlG,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_T,  //we need to recover the "row major"
                                   Y->desc.sizes.w,
                                   X->desc.sizes.w,
                                   X->desc.sizes.h,
                                   &a,
                                   (float*)Y->grad->data,
                                   Y->desc.sizes.h,
                                   (float*)X->data->data,
                                   X->desc.sizes.w,
                                  &b,
                                  (float*)W->grad->data,
                                  W->desc.sizes.w
                                   ))
    }
    
    void HALF_LINEAR(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y){
        //run gemm (linear operation)
        half a = 1.0, b = 1.0;
    
        checkCUBLAS(cublasHgemm(cublasHdlG,
                                   CUBLAS_OP_T,   //row major to column major for weights
                                   CUBLAS_OP_N,   //read the original row major as  column major, auto trans
                                      W->desc.sizes.w,
                                        X->desc.sizes.h,
                                        X->desc.sizes.w,
                                        &a,
                                        (half*)W->data->data,
                                        W->desc.sizes.h,
                                        (half*)X->data->data,
                                        X->desc.sizes.w,
                                        &b,
                                        (half*)Y->data->data,
                                        Y->desc.sizes.w
        ))
    }
    
    void HALF_LINEAR_GRAD_X(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y){
        half a = 1.0, b = 1.0;
    
        checkCUBLAS(cublasHgemm(cublasHdlG,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                W->desc.sizes.h,
                                Y->desc.sizes.h,
                                W->desc.sizes.w,
                                &a,
                                (half*)W->data->data,
                                W->desc.sizes.h,
                                (half*)Y->grad->data,
                                Y->desc.sizes.w,
                                &b,
                                (half*)X->grad->data,
                                X->desc.sizes.w
        ))
    }
    
    void HALF_LINEAR_GRAD_W(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y){
        half a = 1.0, b = 0.0;
    
        checkCUBLAS(cublasHgemm(cublasHdlG,
                                CUBLAS_OP_T,
                                CUBLAS_OP_T,
                                Y->desc.sizes.w,
                                X->desc.sizes.w,
                                X->desc.sizes.h,
                                &a,
                                (half*)Y->grad->data,
                                Y->desc.sizes.h,
                                (half*)X->data->data,
                                X->desc.sizes.w,
                                &b,
                                (half*)W->grad->data,
                                W->desc.sizes.w
        ))
    }
    
    void DOUBLE_LINEAR(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y){
        //run gemm (linear operation)
        double a = 1.0, b = 1.0;
    
        checkCUBLAS(cublasDgemm_v2(cublasHdlG,
                                   CUBLAS_OP_T,   //row major to column major for weights
                                   CUBLAS_OP_N,   //read the original row major as column major, auto trans
                                   W->desc.sizes.w,
                                   X->desc.sizes.h,
                                   X->desc.sizes.w,
                                   &a,
                                   (double*)W->data->data,
                                   W->desc.sizes.h,
                                   (double*)X->data->data,
                                   X->desc.sizes.w,
                                   &b,
                                   (double*)Y->data->data,
                                   Y->desc.sizes.w))
    }
    
    void DOUBLE_LINEAR_GRAD_X(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y){
        double a = 1.0, b = 1.0;
    
        checkCUBLAS(cublasDgemm_v2(cublasHdlG,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_N,
                                   W->desc.sizes.h,
                                   Y->desc.sizes.h,
                                   W->desc.sizes.w,
                                   &a,
                                   (double*)W->data->data,
                                   W->desc.sizes.h,
                                   (double*)Y->grad->data,
                                   Y->desc.sizes.w,
                                   &b,
                                   (double*)X->grad->data,
                                   X->desc.sizes.w
        ))
    }
    
    void DOUBLE_LINEAR_GRAD_W(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y){
        double a = 1.0, b = 0.0;
    
        checkCUBLAS(cublasDgemm_v2(cublasHdlG,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_T,
                                   Y->desc.sizes.w,
                                   X->desc.sizes.w,
                                   X->desc.sizes.h,
                                   &a,
                                   (double*)Y->grad->data,
                                   Y->desc.sizes.h,
                                   (double*)X->data->data,
                                   X->desc.sizes.w,
                                   &b,
                                   (double*)W->grad->data,
                                   W->desc.sizes.w
        ))
    }
    
    cuTensorBase *linearOp(cuTensorBase* W, cuTensorBase* B, cuTensorBase* X, cuTensorBase* Y){
        //set cublas
        checkCUBLAS(cublasSetMathMode(cublasHdlG, CUBLAS_TENSOR_OP_MATH))
        fillBias(B, Y);
        
        //assert same dtye
        assert(W->desc.dType == X->desc.dType
           && W->desc.dType == Y->desc.dType);
        
        //run gemm (linear operation)
        switch (X->desc.dType) {
            case CUDNN_DATA_FLOAT:
                FLOAT_LINEAR(W, X, Y);
                break;
            case CUDNN_DATA_HALF:
                HALF_LINEAR(W, X, Y);
                break;
            case CUDNN_DATA_DOUBLE:
                DOUBLE_LINEAR(W, X, Y);
                break;
            default:
                throw std::runtime_error("unsupported dtype");
        }
        
        return Y;
    }
    
    void GRAD_LINEAR::backward(cuTensorBase *current) {
        checkCUBLAS(cublasSetMathMode(cublasHdlG, CUBLAS_TENSOR_OP_MATH))
    
        //assert same dtype
        assert(current->desc.dType == X->desc.dType
               && current->desc.dType == current->desc.dType);
        
        cout<<"GRAD LINEAR CALLED"<<endl;
        
        //run gradient for features
        switch (X->desc.dType) {
            case CUDNN_DATA_FLOAT:
                FLOAT_LINEAR_GRAD_X(W, X, current);
                break;
            case CUDNN_DATA_HALF:
                HALF_LINEAR_GRAD_X(W, X, current);
                break;
            case CUDNN_DATA_DOUBLE:
                DOUBLE_LINEAR_GRAD_X(W, X, current);
                break;
            default:
                throw std::runtime_error("unsupported dtype");
        }
        
        //run gradients for weights
        switch (X->desc.dType) {
            case CUDNN_DATA_FLOAT:
                FLOAT_LINEAR_GRAD_W(W, X, current);
                break;
            case CUDNN_DATA_HALF:
                HALF_LINEAR_GRAD_W(W, X, current);
                break;
            case CUDNN_DATA_DOUBLE:
                DOUBLE_LINEAR_GRAD_W(W, X, current);
                break;
            default:
                throw std::runtime_error("unsupported dtype");
        }
        
        //run gradients for biases
        cudaMemcpy(B->grad->data, current->grad->data, B->grad->memSize, cudaMemcpyDeviceToHost);
    }
}