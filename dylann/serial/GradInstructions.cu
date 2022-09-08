//
// Created by Dylan on 9/7/2022.
//

#include "GradInstructions.cuh"

namespace dylann {
    void ADD_GRADS::run() {
        addOpGrad((*params)[A], (*params)[B], alpha, beta);
    }
    
    void ADD_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = A;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = B;
        offset += sizeof(TENSOR_PTR);
        *(float*)(file + offset) = alpha;
        offset += sizeof(float);
        *(float*)(file + offset) = beta;
        offset += sizeof(float);
    }
    
    void SCALE_GRADS::run() {
        scaleOpGrad((*params)[A], alpha);
    }
    
    void SCALE_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = A;
        offset += sizeof(TENSOR_PTR);
        *(float*)(file + offset) = alpha;
        offset += sizeof(float);
    }
    
    void LINEAR_GRADS::run() {
        linearOpGrads((*params)[W], (*params)[B], (*params)[X], (*params)[Y]);
    }
    
    void LINEAR_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = W;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = B;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void CONV2D_GRADS::run() {
        conv2dOpGrads((*params)[X], (*params)[W], (*params)[B], (*params)[Y],
                      padH, padW, strideH, strideW, dilationH, dilationW);
    }
    
    void CONV2D_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = W;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = B;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(int*)(file + offset) = strideH;
        offset += sizeof(int);
        *(int*)(file + offset) = strideW;
        offset += sizeof(int);
        *(int*)(file + offset) = padH;
        offset += sizeof(int);
        *(int*)(file + offset) = padW;
        offset += sizeof(int);
        *(int*)(file + offset) = dilationH;
        offset += sizeof(int);
        *(int*)(file + offset) = dilationW;
        offset += sizeof(int);
    }
    
    void MAXPOOL2D_GRADS::run() {
        maxPoolOpGrads((*params)[X], (*params)[Y], kernelH, kernelW, padH, padW, strideH, strideW);
    }
    
    void MAXPOOL2D_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(int*)(file + offset) = kernelH;
        offset += sizeof(int);
        *(int*)(file + offset) = kernelW;
        offset += sizeof(int);
        *(int*)(file + offset) = strideH;
        offset += sizeof(int);
        *(int*)(file + offset) = strideW;
        offset += sizeof(int);
        *(int*)(file + offset) = padH;
        offset += sizeof(int);
        *(int*)(file + offset) = padW;
        offset += sizeof(int);
    }
    
    void AVGPOOL2D_GRADS::run() {
        avgPoolOpGrads((*params)[X], (*params)[Y], kernelH, kernelW, padH, padW, strideH, strideW);
    }
    
    void AVGPOOL2D_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(int*)(file + offset) = kernelH;
        offset += sizeof(int);
        *(int*)(file + offset) = kernelW;
        offset += sizeof(int);
        *(int*)(file + offset) = strideH;
        offset += sizeof(int);
        *(int*)(file + offset) = strideW;
        offset += sizeof(int);
        *(int*)(file + offset) = padH;
        offset += sizeof(int);
        *(int*)(file + offset) = padW;
        offset += sizeof(int);
    }
    
    void SOFTMAX_GRADS::run() {
        softmaxOpGrads((*params)[X], (*params)[Y], step);
    }
    
    void SOFTMAX_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(int*)(file + offset) = step;
        offset += sizeof(int);
    }
    
    void BATCHNORM_GRADS::run() {
        batchnormOpGrads((*params)[X], (*params)[Y], (*params)[mean], (*params)[var]
                ,(*params)[gamma], (*params)[beta], eps, expAvgFactor);
    }
    
    void BATCHNORM_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = mean;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = var;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = gamma;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = beta;
        offset += sizeof(TENSOR_PTR);
        *(float*)(file + offset) = eps;
        offset += sizeof(float);
        *(float*)(file + offset) = expAvgFactor;
        offset += sizeof(float);
    }
    
    void SOFTMAX_LOG_GRADS::run() {
        softmaxLogOpGrads((*params)[X], (*params)[Y], step);
    }
    
    void SOFTMAX_LOG_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(int*)(file + offset) = step;
        offset += sizeof(int);
    }
    
    void CONCAT_CHANNEL_GRADS::run() {
        auto** inputs = (cuTensorBase**)calloc(paramC, sizeof(cuTensorBase*));
        for (int i = 0; i < paramCount; i++) {
            inputs[i] = (*params)[X[i]];
        }
        concatChannelOpGrads((*params)[Y], inputs, paramC);
    }
    
    void CONCAT_CHANNEL_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(int*)(file + offset) = paramC;
        offset += sizeof(int);
        for (int i = 0; i < paramC; i++) {
            *(TENSOR_PTR*)(file + offset) = X[i];
            offset += sizeof(TENSOR_PTR);
        }
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void DROPOUT_GRADS::run() {
        dropoutOpGrads((*params)[X], (*params)[Y], p);
    }
    
    void DROPOUT_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(float*)(file + offset) = p;
        offset += sizeof(float);
    }
    
    void FLATTEN_GRADS::run() {
        flattenOpGrad((*params)[X], (*params)[Y]);
    }
    
    void FLATTEN_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void GLOBAL_AVGPOOL_GRADS::run() {
        globalAvgPoolOpGrads((*params)[X], (*params)[Y]);
    }
    
    void GLOBAL_AVGPOOL_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void SOFTMAX_CE_GRADS::run() {
        softmaxCEOpGrads((*params)[X], (*params)[Y], step);
    }
    
    void SOFTMAX_CE_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(int*)(file + offset) = step;
        offset += sizeof(int);
    }
    
    void RELU_GRADS::run() {
        reluOpGrads((*params)[X], (*params)[Y]);
    }
    
    void RELU_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void SIGMOID_GRADS::run() {
        sigmoidOpGrads((*params)[X], (*params)[Y]);
    }
    
    void SIGMOID_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void TANH_GRADS::run() {
        tanhOpGrads((*params)[X], (*params)[Y]);
    }
    
    void TANH_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void ELU_GRADS::run() {
        eluOpGrads((*params)[X], (*params)[Y], alpha);
    }
    
    void ELU_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(float*)(file + offset) = alpha;
        offset += sizeof(float);
    }
    
    void SWISH_GRADS::run() {
        swishOpGrads((*params)[X], (*params)[Y], beta);
    }
    
    void SWISH_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(float*)(file + offset) = beta;
        offset += sizeof(float);
    }
    
    void CLIPPED_RELU_GRADS::run() {
        clippedReluOpGrads((*params)[X], (*params)[Y], threshold);
    }
    
    void CLIPPED_RELU_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
    
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(float*)(file + offset) = threshold;
        offset += sizeof(float);
    }
    
    //EXTRACTION
    
    ADD_GRADS* extractAddGrads(const unsigned char * file, size_t &offset){
        
        TENSOR_PTR A = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR B = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float alpha = *(float*)(file + offset);
        offset += sizeof(float);
        float beta = *(float*)(file + offset);
        offset += sizeof(float);
    
        return new ADD_GRADS(A, B, alpha, beta);
    }
    
    SCALE_GRADS* extractScaleGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR A = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float alpha = *(float*)(file + offset);
        offset += sizeof(float);
        
        return new SCALE_GRADS(A, alpha);
    }
    
    LINEAR_GRADS* extractLinearGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR W = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR B = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        return new LINEAR_GRADS(W, B, X, Y);
    }
    
    CONV2D_GRADS* extractConv2DGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR W = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR B = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        int strideH = *(int*)(file + offset);
        offset += sizeof(int);
        int strideW = *(int*)(file + offset);
        offset += sizeof(int);
        int padH = *(int*)(file + offset);
        offset += sizeof(int);
        int padW = *(int*)(file + offset);
        offset += sizeof(int);
        int dilationH = *(int*)(file + offset);
        offset += sizeof(int);
        int dilationW = *(int*)(file + offset);
        offset += sizeof(int);
        
        return new CONV2D_GRADS(W, B, X, Y, strideH, strideW, padH, padW, dilationH, dilationW);
    }
    
    MAXPOOL2D_GRADS* extractMaxPool2DGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        int kernelH = *(int*)(file + offset);
        offset += sizeof(int);
        int kernelW = *(int*)(file + offset);
        offset += sizeof(int);
        int strideH = *(int*)(file + offset);
        offset += sizeof(int);
        int strideW = *(int*)(file + offset);
        offset += sizeof(int);
        int padH = *(int*)(file + offset);
        offset += sizeof(int);
        int padW = *(int*)(file + offset);
        offset += sizeof(int);
        
        return new MAXPOOL2D_GRADS(X, Y, kernelH, kernelW, strideH, strideW, padH, padW);
    }
    
    AVGPOOL2D_GRADS* extractAvgPool2DGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        int kernelH = *(int*)(file + offset);
        offset += sizeof(int);
        int kernelW = *(int*)(file + offset);
        offset += sizeof(int);
        int strideH = *(int*)(file + offset);
        offset += sizeof(int);
        int strideW = *(int*)(file + offset);
        offset += sizeof(int);
        int padH = *(int*)(file + offset);
        offset += sizeof(int);
        int padW = *(int*)(file + offset);
        offset += sizeof(int);
        
        return new AVGPOOL2D_GRADS(X, Y, kernelH, kernelW, strideH, strideW, padH, padW);
    }
    
    SOFTMAX_GRADS* extractSoftmaxGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        int step = *(int*)(file + offset);
        offset += sizeof(int);
        
        return new SOFTMAX_GRADS(X, Y, step);
    }
    
    BATCHNORM_GRADS* extractBatchNormGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR mean = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR var = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR gamma = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR beta = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float eps = *(float*)(file + offset);
        offset += sizeof(float);
        float avgExpFactor = *(float*)(file + offset);
        offset += sizeof(float);
        
        return new BATCHNORM_GRADS(X, Y, gamma, beta, mean, var, eps, avgExpFactor);
    }
    
    SOFTMAX_LOG_GRADS* extractSoftmaxLogGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        int step = *(int*)(file + offset);
        offset += sizeof(int);
        
        return new SOFTMAX_LOG_GRADS(X, Y, step);
    }
    
    CONCAT_CHANNEL_GRADS* extractConcatChannelGrads(const unsigned char * file, size_t &offset){
        int num = *(int*)(file + offset);
        offset += sizeof(int);
        TENSOR_PTR * X;
        cudaMallocHost((void**)&X, num * sizeof(TENSOR_PTR));
        for(int i = 0; i < num; i++){
            X[i] = *(TENSOR_PTR*)(file + offset);
            offset += sizeof(TENSOR_PTR);
        }
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        return new CONCAT_CHANNEL_GRADS(X, Y, num);
    }
    
    DROPOUT_GRADS* extractDropoutGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float p = *(float*)(file + offset);
        offset += sizeof(float);
        
        return new DROPOUT_GRADS(X, Y, p);
    }
    
    FLATTEN_GRADS* extractFlattenGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        return new FLATTEN_GRADS(X, Y);
    }
    
    SOFTMAX_CE_GRADS* extractSoftmaxCEGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        int step = *(int*)(file + offset);
        offset += sizeof(int);
        
        return new SOFTMAX_CE_GRADS(X, Y, step);
    }
    
    RELU_GRADS* extractReluGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        return new RELU_GRADS(X, Y);
    }
    
    SIGMOID_GRADS* extractSigmoidGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        return new SIGMOID_GRADS(X, Y);
    }
    
    TANH_GRADS* extractTanhGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        return new TANH_GRADS(X, Y);
    }
    
    ELU_GRADS* extractEluGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float alpha = *(float*)(file + offset);
        offset += sizeof(float);
        
        return new ELU_GRADS(X, Y, alpha);
    }
    
    SWISH_GRADS* extractSwishGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float beta = *(float*)(file + offset);
        offset += sizeof(float);
        
        return new SWISH_GRADS(X, Y, beta);
    }
    
    CLIPPED_RELU_GRADS* extractClippedReluGrads(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float max = *(float*)(file + offset);
        offset += sizeof(float);
        
        return new CLIPPED_RELU_GRADS(X, Y, max);
    }
} // dylann