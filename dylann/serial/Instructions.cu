//
// Created by Dylan on 9/3/2022.
//

#include "Instructions.cuh"

namespace dylann {
    void ADD::run() {
        addOp((*params)[A], (*params)[B], alpha, beta);
    }
    
    void ADD::encodeParams(unsigned char *file,size_t &offset) {
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
    
    void SCALE::run() {
        scale((*params)[A], alpha);
    }
    
    void SCALE::encodeParams(unsigned char *file,size_t &offset) {
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        *(TENSOR_PTR*)(file + offset) = A;
        offset += sizeof(TENSOR_PTR);
        *(float*)(file + offset) = alpha;
        offset += sizeof(float);
    }
    
    void LINEAR::run() {
        linearOp((*params)[W], (*params)[B], (*params)[X], (*params)[Y]);
    }
    
    void LINEAR::encodeParams(unsigned char *file,size_t &offset) {
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
    
    void CONV2D::run() {
        conv2dOp((*params)[X], (*params)[W], (*params)[B], (*params)[Y], padH, padW, strideH, strideW, dilationH, dilationW);
    }
    
    void CONV2D::encodeParams(unsigned char *file,size_t &offset) {
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
    
    
    void MAXPOOL2D::run() {
        maxPoolOp((*params)[X], (*params)[Y], kernelH, kernelW, padH, padW, strideH, strideW);
    }
    
    void MAXPOOL2D::encodeParams(unsigned char *file,size_t &offset) {
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
    
    void AVGPOOL2D::run() {
        avgPoolOp((*params)[X], (*params)[Y], kernelH, kernelW, padH, padW, strideH, strideW);
    }
    
    void AVGPOOL2D::encodeParams(unsigned char *file,size_t &offset) {
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
    
    void GLOBAL_AVGPOOL2D::run() {
        globalAvgPoolOp((*params)[X], (*params)[Y]);
    }
    
    void GLOBAL_AVGPOOL2D::encodeParams(unsigned char *file,size_t &offset) {
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void SOFTMAX::run() {
        softmaxOp((*params)[X], (*params)[Y], step);
    }
    
    void SOFTMAX::encodeParams(unsigned char *file,size_t &offset) {
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
    
    void SOFTMAX_CE::run() {
        softmaxCEOp((*params)[X], (*params)[Y],  step);
    }
    
    void SOFTMAX_CE::encodeParams(unsigned char *file,size_t &offset) {
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
    
    void BATCHNORM::run() {
        if (train) {
            batchnormOp((*params)[X], (*params)[Y], (*params)[mean],
                        (*params)[var], (*params)[gamma], (*params)[beta], eps, expAvgFactor);
        }else{
            batchnormInferOp((*params)[X], (*params)[Y], (*params)[mean],
                             (*params)[var], (*params)[gamma], (*params)[beta], eps);
        }
    }
    
    void BATCHNORM::encodeParams(unsigned char * file, size_t &offset){
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
    
    void SOFTMAX_LOG::run() {
        softmaxLogOp((*params)[X], (*params)[Y], step);
    }
    
    void SOFTMAX_LOG::encodeParams(unsigned char *file,size_t &offset) {
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
    
    void CONCAT_CHANNEL::run() {
        auto** inputs = (cuTensorBase**)calloc(paramC, sizeof(cuTensorBase*));
        for (int i = 0; i < paramCount; i++) {
            inputs[i] = (*params)[X[i]];
        }
        concatChannelOp(inputs, paramCount, (*params)[Y]);
    }
    
    void CONCAT_CHANNEL::encodeParams(unsigned char *file,size_t &offset) {
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
    
    void DROPOUT::run() {
        dropoutOp((*params)[X], (*params)[Y], rate);
    }
    
    void DROPOUT::encodeParams(unsigned char *file,size_t &offset) {
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(float*)(file + offset) = rate;
        offset += sizeof(float);
    }
    
    void FLATTEN::run() {
        flattenOp((*params)[X], (*params)[Y]);
    }
    
    void FLATTEN::encodeParams(unsigned char *file,size_t &offset) {
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void RELU::run() {
        reluOp((*params)[X], (*params)[Y]);
    }
    
    void RELU::encodeParams(unsigned char *file,size_t &offset) {
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void SIGMOID::run() {
        sigmoidOp((*params)[X], (*params)[Y]);
    }
    
    void SIGMOID::encodeParams(unsigned char *file,size_t &offset) {
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void TANH::run() {
        tanhOp((*params)[X], (*params)[Y]);
    }
    
    void TANH::encodeParams(unsigned char *file,size_t &offset) {
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void ELU::run() {
        eluOp((*params)[X], (*params)[Y], alpha);
    }
    
    void ELU::encodeParams(unsigned char *file,size_t &offset) {
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
    
    void SWISH::run() {
        swishOp((*params)[X], (*params)[Y], beta);
    }
    
    void SWISH::encodeParams(unsigned char *file,size_t &offset) {
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
    
    void CLIPPED_RELU::run() {
        clippedReluOp((*params)[X], (*params)[Y], threshold);
    }
    
    void CLIPPED_RELU::encodeParams(unsigned char *file,size_t &offset) {
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
    
    
    
    //EXTRACT
    //------------------------------------------------------
    
    ADD* extractAdd(const unsigned char * file, size_t &offset) {
        TENSOR_PTR A = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR B = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float alpha = *(float*)(file + offset);
        offset += sizeof(float);
        float beta = *(float*)(file + offset);
        offset += sizeof(float);
        
        auto* add = new ADD(A, B, alpha, beta);
        return add;
    }
    
    SCALE* extractScale(const unsigned char * file, size_t &offset) {
        TENSOR_PTR A = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float alpha = *(float*)(file + offset);
        offset += sizeof(float);
        
        auto* scale = new SCALE(A, alpha);
        return scale;
    }
    
    LINEAR* extractLinear(const unsigned char * file, size_t &offset) {
        TENSOR_PTR W = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR B = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        auto* linear = new LINEAR(W, B, X, Y);
        return linear;
    }
    
    CONV2D* extractConv2D(const unsigned char * file, size_t &offset){
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
        
        auto* conv2d = new CONV2D(W, B, X, Y, strideH, strideW, padH, padW, dilationH, dilationW);
        return conv2d;
    }
    
    MAXPOOL2D* extractMaxPool2D(const unsigned char * file, size_t &offset){
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
        
        auto* maxPool2d = new MAXPOOL2D(X, Y, kernelH, kernelW, strideH, strideW, padH, padW);
        return maxPool2d;
    }
    
    AVGPOOL2D* extractAvgPool2D(const unsigned char * file, size_t &offset){
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
        
        auto* avgPool2d = new AVGPOOL2D(X, Y, kernelH, kernelW, strideH, strideW, padH, padW);
        return avgPool2d;
    }
    
    GLOBAL_AVGPOOL2D* extractGlobalAvgPool2D(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        auto* globalAvgPool2d = new GLOBAL_AVGPOOL2D(X, Y);
        return globalAvgPool2d;
    }
    
    SOFTMAX* extractSoftmax(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        int axis = *(int*)(file + offset);
        offset += sizeof(int);
        
        auto* softmax = new SOFTMAX(X, Y, axis);
        return softmax;
    }
    
    SOFTMAX_CE* extractSoftmaxCE(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        int axis = *(int*)(file + offset);
        offset += sizeof(int);
        
        auto* softmaxCE = new SOFTMAX_CE(X, Y, axis);
        return softmaxCE;
    }
    
    BATCHNORM* extractBatchNorm(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR mean = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR var = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR weight = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR bias = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float eps = *(float*)(file + offset);
        offset += sizeof(float);
        float expAvgFactor = *(float*)(file + offset);
        offset += sizeof(float);
        
        auto* batchNorm = new BATCHNORM(X, Y, weight, bias, mean, var, eps, expAvgFactor);
        return batchNorm;
    }
    
    SOFTMAX_LOG* extractSoftmaxLog(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        int axis = *(int*)(file + offset);
        offset += sizeof(int);
        
        auto* softmaxLog = new SOFTMAX_LOG(X, Y, axis);
        return softmaxLog;
    }
    
    CONCAT_CHANNEL* extractConcatChannel(const unsigned char * file, size_t &offset){
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
        
        auto* concatChannel = new CONCAT_CHANNEL(X, Y, num);
        return concatChannel;
    }
    
    DROPOUT* extractDropout(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float ratio = *(float*)(file + offset);
        offset += sizeof(float);
        
        auto* dropout = new DROPOUT(X, Y, ratio);
        return dropout;
    }
    
    FLATTEN* extractFlatten(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        auto* flatten = new FLATTEN(X, Y);
        return flatten;
    }
    
    RELU* extractRelu(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        auto* relu = new RELU(X, Y);
        return relu;
    }
    
    SIGMOID* extractSigmoid(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        auto* sigmoid = new SIGMOID(X, Y);
        return sigmoid;
    }
    
    TANH* extractTanh(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        auto* tanh = new TANH(X, Y);
        return tanh;
    }
    
    ELU* extractElu(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float alpha = *(float*)(file + offset);
        offset += sizeof(float);
        
        auto* elu = new ELU(X, Y, alpha);
        return elu;
    }
    
    SWISH* extractSwish(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float beta = *(float*)(file + offset);
        offset += sizeof(float);
        
        auto* swish = new SWISH(X, Y, beta);
        return swish;
    }
    
    CLIPPED_RELU* extractClippedRelu(const unsigned char * file, size_t &offset){
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float threshold = *(float*)(file + offset);
        offset += sizeof(float);
        
        auto* clippedRelu = new CLIPPED_RELU(X, Y, threshold);
        return clippedRelu;
    }
} // dylann