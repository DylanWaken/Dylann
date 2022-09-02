//
// Created by Dylan on 8/31/2022.
//

#include "cuConv.cuh"
namespace dylann{
    cuTensorBase* conv2dOp(cuTensorBase* X, cuTensorBase* W, cuTensorBase* B, cuTensorBase* Y,
                           int padH, int padW, int strideH, int strideW, int dilationH, int dilationW){
        assertAllocated({W, B, X, Y});
        assertOnSameDev({W, B, X, Y});
        
        cudnnConvolutionDescriptor_t convDesc;
        cudnnCreateConvolutionDescriptor(&convDesc);
        cudnnFilterDescriptor_t filterDesc;
        cudnnCreateFilterDescriptor(&filterDesc);
        cudnnActivationDescriptor_t activationDesc;
        cudnnCreateActivationDescriptor(&activationDesc);
    
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strideH, strideW, dilationH, dilationW,
                                        CUDNN_CROSS_CORRELATION, X->desc.dType))
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, X->desc.dType,
                                   CUDNN_TENSOR_NCHW,
                                   (int)W->desc.sizes.n,
                                   (int)W->desc.sizes.c,
                                   (int)W->desc.sizes.h,
                                   (int)W->desc.sizes.w
                                   ))
                                   
        checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0))
    
        checkCUDNN(cudnnSetConvolutionMathType(convDesc,  CUDNN_TENSOR_OP_MATH))
    
        float alpha = 1.0f, alpha2 = 0.0f;

        checkCUDNN(cudnnConvolutionBiasActivationForward(
                cudnnHdlG,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                filterDesc,
                W->data->data,
                convDesc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                cudnnWorkspaceG,
                CUDNN_WORKSPACE_SIZE_G,
                &alpha2,
                Y->desc.cudnnDesc,
                Y->data->data,
                B->desc.cudnnDesc,
                B->data->data,
                activationDesc,
                Y->desc.cudnnDesc,
                Y->data->data
                ))

        cudnnDestroyActivationDescriptor(activationDesc);
        cudnnDestroyFilterDescriptor(filterDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudaMemset(cudnnWorkspaceG, 0, CUDNN_WORKSPACE_SIZE_G);
    
        assertCuda(__FILE__, __LINE__);
        return Y;
    }
    
    cuTensorBase* conv2dActiveOp(cuTensorBase* X, cuTensorBase* W, cuTensorBase* B, cuTensorBase* Y,
                                 int padH, int padW, int strideH, int strideW, int dilationH, int dilationW,
                                 cudnnActivationMode_t mode, float coef){
        assertAllocated({W, B, X, Y});
        assertOnSameDev({W, B, X, Y});
    
        cudnnConvolutionDescriptor_t convDesc;
        cudnnCreateConvolutionDescriptor(&convDesc);
        cudnnFilterDescriptor_t filterDesc;
        cudnnCreateFilterDescriptor(&filterDesc);
        cudnnActivationDescriptor_t activationDesc;
        cudnnCreateActivationDescriptor(&activationDesc);
    
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strideH, strideW, dilationH, dilationW,
                                                   CUDNN_CROSS_CORRELATION, X->desc.dType))
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, X->desc.dType,
                                              CUDNN_TENSOR_NCHW,
                                              (int)W->desc.sizes.n,
                                              (int)W->desc.sizes.c,
                                              (int)W->desc.sizes.h,
                                              (int)W->desc.sizes.w))
        checkCUDNN(cudnnSetActivationDescriptor(activationDesc, mode, CUDNN_NOT_PROPAGATE_NAN, coef))
    
        checkCUDNN(cudnnSetConvolutionMathType(convDesc,  CUDNN_TENSOR_OP_MATH))
    
        float alpha = 1.0f, alpha2 = 0.0f;
    
        checkCUDNN(cudnnConvolutionBiasActivationForward(
                cudnnHdlG,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                filterDesc,
                W->data->data,
                convDesc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                cudnnWorkspaceG,
                CUDNN_WORKSPACE_SIZE_G,
                &alpha2,
                Y->desc.cudnnDesc,
                Y->data->data,
                B->desc.cudnnDesc,
                B->data->data,
                activationDesc,
                Y->desc.cudnnDesc,
                Y->data->data
        ))
    
        cudnnDestroyActivationDescriptor(activationDesc);
        cudnnDestroyFilterDescriptor(filterDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudaMemset(cudnnWorkspaceG, 0, CUDNN_WORKSPACE_SIZE_G);
        
        return Y;
    }
    
    void GRAD_CONV2D::backwardCalc(cuTensorBase *Y) {
        cudnnConvolutionDescriptor_t convDesc;
        cudnnCreateConvolutionDescriptor(&convDesc);
        
        cudnnFilterDescriptor_t filterDesc;
        cudnnCreateFilterDescriptor(&filterDesc);
        
        float alpha = 1.0f, beta = 1.0f;
    
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strideH, strideW, dilationH, dilationW,
                                                   CUDNN_CROSS_CORRELATION, Y->desc.dType))
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, Y->desc.dType,
                                              CUDNN_TENSOR_NCHW,
                                              (int)W->desc.sizes.n,
                                              (int)W->desc.sizes.c,
                                              (int)W->desc.sizes.h,
                                              (int)W->desc.sizes.w));
        
        checkCUDNN(cudnnSetConvolutionMathType(convDesc,  CUDNN_TENSOR_OP_MATH))
        
        checkCUDNN(cudnnConvolutionBackwardData(cudnnHdlG,
                &alpha,
                filterDesc,
                W->data->data,
                Y->desc.cudnnDesc,
                Y->grad->data,
                convDesc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                nullptr,
                0,
                &beta,
                X->desc.cudnnDesc,
                X->grad->data
        ))
        
        checkCUDNN(cudnnConvolutionBackwardFilter(
                cudnnHdlG,
                &alpha,
                X->desc.cudnnDesc,
                X->data->data,
                Y->desc.cudnnDesc,
                Y->grad->data,
                convDesc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                nullptr,
                0,
                &beta,
                filterDesc,
                W->grad->data
        ))
    
        checkCUDNN(cudnnConvolutionBackwardBias(
                cudnnHdlG,
                &alpha,
                Y->desc.cudnnDesc,
                Y->grad->data,
                &beta,
                B->desc.cudnnDesc,
                B->grad->data
        ))
        
        cudnnDestroyFilterDescriptor(filterDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
    }
}
