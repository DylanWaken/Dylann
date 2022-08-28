//
// Created by Dylan on 8/25/2022.
//

#include "cuLinear.cuh"

namespace dylann{
    
    void FLOAT_GEMM(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y){
        //run gemm (linear operation)
        float a = 1.0f, b = 0.0f;
    
        checkCUBLAS(cublasSgemm_v2(cublasHdlG,
                                   CUBLAS_OP_T,   //row major to column major for weights
                                   CUBLAS_OP_N,   //read the original row major as column major, auto trans
                                   W->desc.sizes.w,
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
    
    cuTensorBase *linearOp(cuTensorBase* W, cuTensorBase* X, cuTensorBase* Y){
        //emplace the biases into the model
        
        //set cublas
        checkCUBLAS(cublasSetMathMode(cublasHdlG, CUBLAS_TENSOR_OP_MATH))
        
        //assert same dtye
        assert(W->desc.dType == X->desc.dType
           && W->desc.dType == Y->desc.dType);
        
        //run gemm (linear operation)
        switch (X->desc.dType) {
            case CUDNN_DATA_FLOAT:
                FLOAT_GEMM(W, X, Y);
                break;
            default:
                throw std::runtime_error("unsupported dtype");
        }
        
        return Y;
    }
}