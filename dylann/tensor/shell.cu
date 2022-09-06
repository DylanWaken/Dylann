//
// Created by Dylan on 8/8/2022.
//

#include "shell.cuh"

namespace dylann{
    cuTensor add(cuTensor& A, cuTensor& B, float alpha, float beta){
        addOp(A.impl, B.impl, alpha, beta);
        return A;
    }
    
    cuTensor scale(cuTensor& A, float alpha){
        scale(A.impl, alpha);
        return A;
    }
    
    cuTensor linear(cuTensor& W, cuTensor& B, cuTensor& X, cuTensor& Y){
        linearOp(W.impl, B.impl, X.impl, Y.impl);
        return Y;
    }
    
    cuTensor conv2D(cuTensor& X, cuTensor& W, cuTensor& B, cuTensor& Y,
                    int padH, int padW, int strideH, int strideW, int dilationH, int dilationW){
        conv2dOp(X.impl, W.impl, B.impl, Y.impl, padH, padW, strideH, strideW, dilationH, dilationW);
        return Y;
    }
    
    cuTensor reduce(cuTensor& X, cuTensor& Y, int step){
        reduceOp(X.impl, Y.impl, step);
        return Y;
    }
    
    cuTensor softmax(cuTensor& X, cuTensor& Y, int step){
        softmaxOp(X.impl, Y.impl, step);
        return Y;
    }
    
    cuTensor softmaxLog(cuTensor& X, cuTensor& Y, int step){
        softmaxLogOp(X.impl, Y.impl, step);
        return Y;
    }
    
    cuTensor softmaxCE(cuTensor& X, cuTensor& Y, int step){
        softmaxCEOp(X.impl, Y.impl, step);
        return Y;
    }
    
    cuTensor channelConcat(cuTensor* Xs, int inputCount, cuTensor& Y, cuTensor& XGradTarget){
        cuTensorBase** XsImpl;
        cudaMallocHost(&XsImpl, sizeof(cuTensorBase*) * inputCount);
        for(int i = 0; i < inputCount; i++){
            XsImpl[i] = Xs[i].impl;
        }
        concatChannelOp(XsImpl, inputCount, Y.impl);
        return Y;
    }
    
    cuTensor channelConcat(std::initializer_list<cuTensor> inputs, cuTensor& Y, cuTensor& XGradTarget){
        cuTensorBase** XsImpl;
        cudaMallocHost(&XsImpl, sizeof(cuTensorBase*) * inputs.size());
        for(int i = 0; i < inputs.size(); i++){
            XsImpl[i] = inputs.begin()[i].impl;
        }
        concatChannelOp(XsImpl, (int) inputs.size(), Y.impl);
        return Y;
    }
    
    //--------------------------------------------------------------------------------
    //Activations
    
    cuTensor relu(cuTensor& X){
        reluOp(X.impl);
        return X;
    }
    
    cuTensor relu(cuTensor& X, cuTensor& Y){
        reluOp(X.impl, Y.impl);
        return Y;
    }
    
    cuTensor sigmoid(cuTensor& X){
        sigmoidOp(X.impl);
        return X;
    }
    
    cuTensor sigmoid(cuTensor& X, cuTensor& Y){
        sigmoidOp(X.impl, Y.impl);
        return Y;
    }
    
    cuTensor tanh(cuTensor& X){
        tanhOp(X.impl);
        return X;
    }
    
    cuTensor tanh(cuTensor& X, cuTensor& Y){
        tanhOp(X.impl, Y.impl);
        X.impl->desc.gradSrcUuid = Y.desc().uuid;
        
        return Y;
    }
    
    cuTensor elu(cuTensor& X, float alpha){
        eluOp(X.impl, alpha);
        return X;
    }
    
    cuTensor elu(cuTensor& X, cuTensor& Y, float alpha){
        eluOp(X.impl, Y.impl, alpha);
        return Y;
    }
    
    cuTensor swish(cuTensor& X, float beta){
        swishOp(X.impl, beta);
        return X;
    }
    
    cuTensor swish(cuTensor& X, cuTensor& Y, float beta){
        swishOp(X.impl, Y.impl, beta);
        return Y;
    }
    
    cuTensor clippedRelu(cuTensor& X, float threshold){
        clippedReluOp(X.impl, threshold);
        return X;
    }
    
    cuTensor clippedRelu(cuTensor& X, cuTensor& Y, float threshold){
        clippedReluOp(X.impl, Y.impl, threshold);
        return Y;
    }
    
    cuTensor randUniform(cuTensor& A, double min, double max){
        return A.randUniform(min, max);
    }
    
    cuTensor randNormal(cuTensor& A, double mean, double stddev){
        return A.randNormal(mean, stddev);
    }
}
