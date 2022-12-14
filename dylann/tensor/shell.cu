//
// Created by Dylan on 8/8/2022.
//

#include "shell.cuh"
#include <chrono>

namespace dylann{
    cuTensor add(cuTensor X1, cuTensor X2, float alpha, float beta){
        cuTensor Y = cuTensor::create(X1.data()->deviceID, X1.desc().dType, X1.desc().sizes);
        return add(X1, X2, Y, alpha, beta);
    }
    
    cuTensor add(cuTensor X1, cuTensor X2, cuTensor Y, float alpha, float beta){
        addOp(X1.impl, X2.impl, Y.impl, alpha, beta);
    
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new ADD(X1.desc().uuid, X2.desc().uuid, Y.desc().uuid, alpha, beta);
            forwardOpsCTX.push_back(inst);
        }
        return X1;
    }
    
    cuTensor scale(cuTensor X, float alpha){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes);
        return scale(X, Y, alpha);
    }
    
    cuTensor scale(cuTensor X, cuTensor Y, float alpha){
        scale(X.impl, Y.impl, alpha);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new SCALE(X.desc().uuid, Y.desc().uuid, alpha);
            forwardOpsCTX.push_back(inst);
        }
        return X;
    }
    
    cuTensor linear(cuTensor W, cuTensor B, cuTensor X, cuTensor Y, float alpha1, float alpha2){
        linearOp(W.impl, B.impl, X.impl, Y.impl, alpha1, alpha2);
    
        W.asNetworkParam().asNetworkWeight();
        B.asNetworkParam();
        W.impl->setInitType(INIT_XAVIER_LINEAR_WEIGHT);
        B.impl->setInitType(INIT_XAVIER_LINEAR_BIAS);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new LINEAR(W.desc().uuid, B.desc().uuid, X.desc().uuid, Y.desc().uuid, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor linear(cuTensor W, cuTensor B, cuTensor X, cuTensor Y){
        linearOp(W.impl, B.impl, X.impl, Y.impl, 1, 0);
    
        W.impl->setInitType(INIT_XAVIER_LINEAR_WEIGHT);
        B.impl->setInitType(INIT_XAVIER_LINEAR_BIAS);
        
        W.asNetworkParam().asNetworkWeight();
        B.asNetworkParam();
        
        W.asNetworkParam().asNetworkWeight();
        B.asNetworkParam();
        W.impl->setInitType(INIT_XAVIER_LINEAR_WEIGHT);
        B.impl->setInitType(INIT_XAVIER_LINEAR_BIAS);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new LINEAR(W.desc().uuid, B.desc().uuid, X.desc().uuid, Y.desc().uuid ,1, 0);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor linear(cuTensor W, cuTensor B, cuTensor X){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n, 1, 1, W.desc().sizes.h);
        return linear(W, B, X, Y);
    }
    
    cuTensor linear(cuTensor X, int outDim){
        cuTensor W = cuTensor::create(X.data()->deviceID, X.desc().dType, 1, 1, outDim, X.desc().sizes.w);
        cuTensor B = cuTensor::create(X.data()->deviceID, X.desc().dType, 1, 1, 1, outDim);
        return linear(W, B, X);
    }
    
    cuTensor conv2D(cuTensor X, cuTensor W, cuTensor B, cuTensor Y,
                      int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, float alpha1, float alpha2){
        conv2dOp(X.impl, W.impl, B.impl, Y.impl,  strideH, strideW,padH, padW, dilationH, dilationW, alpha1, alpha2);
        
        W.asNetworkParam().asNetworkWeight();
        B.asNetworkParam();
    
        W.impl->setInitType(INIT_STD_CONV_WEIGHT);
        B.impl->setInitType(INIT_STD_CONV_BIAS);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new CONV2D(X.desc().uuid, W.desc().uuid, B.desc().uuid, Y.desc().uuid,
                                    strideH, strideW, padH, padW, dilationH, dilationW, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor conv2D(cuTensor X, cuTensor W, cuTensor B, cuTensor Y,
                    int strideH, int strideW, int padH, int padW, int dilationH, int dilationW){
        conv2dOp(X.impl, W.impl, B.impl, Y.impl, strideH, strideW, padH, padW, dilationH, dilationW, 1, 0);
    
        W.impl->setInitType(INIT_STD_CONV_WEIGHT);
        B.impl->setInitType(INIT_STD_CONV_BIAS);
        
        W.asNetworkParam().asNetworkWeight();
        B.asNetworkParam();
        
        W.asNetworkParam().asNetworkWeight();
        B.asNetworkParam();
        
        W.impl->setInitType(INIT_STD_CONV_WEIGHT);
        B.impl->setInitType(INIT_STD_CONV_BIAS);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new CONV2D(W.desc().uuid, B.desc().uuid, X.desc().uuid,  Y.desc().uuid,
                                    strideH, strideW, padH, padW, dilationH, dilationW, 1, 0);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor conv2D(cuTensor X, cuTensor W, cuTensor B,
                    int strideH, int strideW, int padH, int padW,int dilationH, int dilationW){
        unsigned int n = X.desc().sizes.n;
        unsigned int c = W.desc().sizes.n;
        unsigned int h = (X.desc().sizes.h + 2*padH - (dilationH*(W.desc().sizes.h-1) + 1))/strideH + 1;
        unsigned int w = (X.desc().sizes.w + 2*padW - (dilationW*(W.desc().sizes.w-1) + 1))/strideW + 1;
        
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, n, c, h, w);
        return conv2D(X, W, B, Y,  strideH, strideW,padH, padW, dilationH, dilationW);
    }
    
    cuTensor conv2D(cuTensor X, int kernelH, int kernelW, int outChannels,
                    int strideH, int strideW, int padH, int padW, int dilationH, int dilationW){
        unsigned int Wn = outChannels;
        unsigned int Wc = X.desc().sizes.c;
        unsigned int Wh = kernelH;
        unsigned int Ww = kernelW;
        
        unsigned int Bn = 1;
        unsigned int Bc = outChannels;
        unsigned int Bh = 1;
        unsigned int Bw = 1;
        
        cuTensor W = cuTensor::create(X.data()->deviceID, X.desc().dType, Wn, Wc, Wh, Ww);
        cuTensor B = cuTensor::create(X.data()->deviceID, X.desc().dType, Bn, Bc, Bh, Bw);
        
        return conv2D(X, W, B, strideH, strideW, padH, padW,  dilationH, dilationW);
    }
    
    
    cuTensor reduce(cuTensor X, cuTensor Y, int step){
        reduceOp(X.impl, Y.impl, step);
        return Y;
    }
    
    cuTensor softmax(cuTensor X, cuTensor Y, int step){
        softmaxOp(X.impl, Y.impl, step);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new SOFTMAX(X.desc().uuid, Y.desc().uuid, step);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor softmax(cuTensor X, int step){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return softmax(X, Y, step);
    }
    
    cuTensor softmaxLog(cuTensor X, cuTensor Y, int step){
        softmaxLogOp(X.impl, Y.impl, step);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new SOFTMAX_LOG(X.desc().uuid, Y.desc().uuid, step);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor softmaxLog(cuTensor X, int step){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return softmaxLog(X, Y, step);
    }
    
    cuTensor softmaxCE(cuTensor X, cuTensor Y, int step){
        softmaxCEOp(X.impl, Y.impl, step);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new SOFTMAX_CE(X.desc().uuid, Y.desc().uuid, step);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor softmaxCE(cuTensor X, int step){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return softmaxCE(X, Y, step);
    }
    
    cuTensor channelConcat(cuTensor* Xs, int inputCount, cuTensor Y){
        cuTensorBase** XsImpl;
        cudaMallocHost(&XsImpl, sizeof(cuTensorBase*) * inputCount);
        for(int i = 0; i < inputCount; i++){
            XsImpl[i] = Xs[i].impl;
        }
        concatChannelOp(XsImpl, inputCount, Y.impl);
    
        if(regisModeCTX){
            TENSOR_PTR* XsPtr;
            cudaMallocHost(&XsPtr, sizeof(TENSOR_PTR) * inputCount);
            
            for(int i = 0; i < inputCount; i++){
                XsPtr[i] = Xs[i].desc().uuid;
            }
    
            //push forward instruction
            auto* inst = new CONCAT_CHANNEL(XsPtr, Y.desc().uuid, inputCount);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor channelConcat(cuTensor* Xs, int inputCount){
        unsigned int n = Xs[0].desc().sizes.n;
        unsigned int c = 0;
        unsigned int h = Xs[0].desc().sizes.h;
        unsigned int w = Xs[0].desc().sizes.w;
        for(int i = 0; i < inputCount; i++){
            c += Xs[i].desc().sizes.c;
        }
        cuTensor Y = cuTensor::create(Xs[0].data()->deviceID, Xs[0].desc().dType, n, c, h, w);
        return channelConcat(Xs, inputCount, Y);
    }
    
    cuTensor channelConcat(std::initializer_list<cuTensor> inputs, cuTensor Y){
        cuTensorBase** XsImpl;
        cudaMallocHost(&XsImpl, sizeof(cuTensorBase*) * inputs.size());
        for(int i = 0; i < inputs.size(); i++){
            XsImpl[i] = inputs.begin()[i].impl;
        }
        concatChannelOp(XsImpl, (int) inputs.size(), Y.impl);
    
        if(regisModeCTX){
            TENSOR_PTR* XsPtr;
            cudaMallocHost(&XsPtr, sizeof(TENSOR_PTR) * inputs.size());
        
            for(int i = 0; i < inputs.size(); i++){
                XsPtr[i] = inputs.begin()[i].desc().uuid;
            }
        
            //push forward instruction
            auto* inst = new CONCAT_CHANNEL(XsPtr, Y.desc().uuid, inputs.size());
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor channelConcat(std::initializer_list<cuTensor> inputs){
        unsigned int n = inputs.begin()[0].desc().sizes.n;
        unsigned int c = 0;
        unsigned int h = inputs.begin()[0].desc().sizes.h;
        unsigned int w = inputs.begin()[0].desc().sizes.w;
        for(int i = 0; i < inputs.size(); i++){
            c += inputs.begin()[i].desc().sizes.c;
        }
        cuTensor Y = cuTensor::create(inputs.begin()[0].data()->deviceID, inputs.begin()[0].desc().dType, n, c, h, w);
        return channelConcat(inputs, Y);
    }
    
    cuTensor batchnorm(cuTensor X, cuTensor Y, cuTensor runningMean, cuTensor runningVar,
                       cuTensor gamma, cuTensor beta, float eps, float expAvgFactor){

        return batchnorm(X, Y, runningMean, runningVar, gamma, beta, eps, expAvgFactor, 1, 0);
    }
    
    cuTensor batchnorm(cuTensor X, cuTensor Y, cuTensor runningMean, cuTensor runningVar,
                       cuTensor gamma, cuTensor beta, float eps, float expAvgFactor, float alpha1, float alpha2){
        batchnormOp(X.impl, Y.impl, runningMean.impl, runningVar.impl,
                    gamma.impl, beta.impl, eps, expAvgFactor, alpha1, alpha2);
    
        gamma.asNetworkParam().asNetworkWeight();
        beta.asNetworkParam();
        gamma.impl->setInitType(INIT_STD_BN_WEIGHT);
        beta.impl->setInitType(INIT_STD_BN_BIAS);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new BATCHNORM(X.desc().uuid, Y.desc().uuid, gamma.desc().uuid, beta.desc().uuid,
                                       runningMean.desc().uuid, runningVar.desc().uuid, eps, expAvgFactor, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        
        return Y;
    }
    
    cuTensor batchnorm(cuTensor X, cuTensor runningMean, cuTensor runningVar,
                       cuTensor gamma, cuTensor beta, float eps, float expAvgFactor){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return batchnorm(X, Y, runningMean, runningVar, gamma, beta, eps, expAvgFactor);
    }
    
    cuTensor batchnorm(cuTensor X, float eps, float expAvgFactor){
        cuTensor runningMean = cuTensor::create(X.data()->deviceID, X.desc().dType,
                                                1, X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);

        cuTensor runningVar = cuTensor::create(X.data()->deviceID, X.desc().dType,
                                                  1, X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
                                                  
        cuTensor gamma = cuTensor::create(X.data()->deviceID, X.desc().dType,
                                            1, X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        cuTensor beta = cuTensor::create(X.data()->deviceID, X.desc().dType,
                                             1, X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        
        return batchnorm(X, runningMean, runningVar, gamma, beta, eps, expAvgFactor);
    }
    
    cuTensor batchnorm2d(cuTensor X, cuTensor Y, cuTensor runningMean, cuTensor runningVar,
                         cuTensor gamma, cuTensor beta, float eps, float expAvgFactor, float alpha1, float alpha2){
        batchnorm2dOp(X.impl, Y.impl, runningMean.impl, runningVar.impl,
                    gamma.impl, beta.impl, eps, expAvgFactor, alpha1, alpha2);
        
        gamma.asNetworkParam().asNetworkWeight();
        beta.asNetworkParam();
        gamma.impl->setInitType(INIT_STD_BN_WEIGHT);
        beta.impl->setInitType(INIT_STD_BN_BIAS);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new BATCHNORM2D(X.desc().uuid, Y.desc().uuid, gamma.desc().uuid, beta.desc().uuid,
                                       runningMean.desc().uuid, runningVar.desc().uuid, eps, expAvgFactor, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        
        return Y;
    }
    
    cuTensor batchnorm2d(cuTensor X, cuTensor Y, cuTensor runningMean, cuTensor runningVar,
                         cuTensor gamma, cuTensor beta, float eps, float expAvgFactor){
        return batchnorm2d(X, Y, runningMean, runningVar, gamma, beta, eps, expAvgFactor, 1, 0);
    }
    
    cuTensor batchnorm2d(cuTensor X, cuTensor runningMean, cuTensor runningVar,
                         cuTensor gamma, cuTensor beta, float eps, float expAvgFactor){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return batchnorm2d(X, Y, runningMean, runningVar, gamma, beta, eps, expAvgFactor);
    }
    
    cuTensor batchnorm2d(cuTensor X, float eps, float expAvgFactor){
        cuTensor runningMean = cuTensor::create(X.data()->deviceID, X.desc().dType,
                                                1, X.desc().sizes.c, 1, 1);
    
        cuTensor runningVar = cuTensor::create(X.data()->deviceID, X.desc().dType,
                                               1, X.desc().sizes.c, 1, 1);
    
        cuTensor gamma = cuTensor::create(X.data()->deviceID, X.desc().dType,
                                          1, X.desc().sizes.c, 1, 1);
    
        cuTensor beta = cuTensor::create(X.data()->deviceID, X.desc().dType,
                                         1, X.desc().sizes.c, 1, 1);
    
        return batchnorm2d(X, runningMean, runningVar, gamma, beta, eps, expAvgFactor);
    }
    
    cuTensor dropout(cuTensor X, cuTensor Y, cuTensor mask, float p){
        dropoutOp(X.impl, Y.impl, mask.impl, p);
    
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new DROPOUT(X.desc().uuid, Y.desc().uuid, mask.desc().uuid, p);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor dropout(cuTensor X, cuTensor Y, float p){
        size_t maskSize;
        cudnnDropoutGetReserveSpaceSize(X.cudnnDesc(), &maskSize);
        size_t stateSize;
        cudnnDropoutGetStatesSize(cudnnHdlG, &stateSize);
        
        cuTensor mask = cuTensor::declare(CUDNN_DATA_INT8, maskSize + stateSize).instantiateData(X.data()->deviceID);
        return dropout(X, Y, mask, p);
    }
    
    cuTensor dropout(cuTensor X, float p){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return dropout(X, Y, p);
    }
    
    cuTensor maxPool2D(cuTensor X, cuTensor Y, int kernelH, int kernelW, int padH, int padW, int strideH, int strideW){
        maxPoolOp(X.impl, Y.impl, kernelH, kernelW, padH, padW, strideH, strideW, 1, 1);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new MAXPOOL2D(X.desc().uuid, Y.desc().uuid, kernelH, kernelW, strideH, strideW, padH, padW,1 ,1);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor maxPool2D(cuTensor X, cuTensor Y, int kernelH, int kernelW, int padH, int padW, int strideH, int strideW,
                       float alpha1, float alpha2){
        maxPoolOp(X.impl, Y.impl, kernelH, kernelW, padH, padW, strideH, strideW, alpha1, alpha2);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new MAXPOOL2D(X.desc().uuid, Y.desc().uuid, kernelH, kernelW, strideH, strideW, padH, padW, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor maxPool2D(cuTensor X, int kernelH, int kernelW, int padH, int padW, int strideH, int strideW){
        unsigned int n = X.desc().sizes.n;
        unsigned int c = X.desc().sizes.c;
        unsigned int h = (X.desc().sizes.h + 2 * padH - kernelH) / strideH + 1;
        unsigned int w = (X.desc().sizes.w + 2 * padW - kernelW) / strideW + 1;
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, n, c, h, w);
        return maxPool2D(X, Y, kernelH, kernelW, padH, padW, strideH, strideW);
    }
    
    cuTensor avgPool2D(cuTensor X, cuTensor Y, int kernelH, int kernelW, int padH, int padW, int strideH, int strideW){
        avgPoolOp(X.impl, Y.impl, kernelH, kernelW, padH, padW, strideH, strideW, 1, 1);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new AVGPOOL2D(X.desc().uuid, Y.desc().uuid, kernelH, kernelW, strideH, strideW, padH, padW, 1, 1);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor avgPool2D(cuTensor X, cuTensor Y, int kernelH, int kernelW, int padH, int padW, int strideH, int strideW,
                       float alpha1, float alpha2){
        avgPoolOp(X.impl, Y.impl, kernelH, kernelW, padH, padW, strideH, strideW, alpha1, alpha2);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new AVGPOOL2D(X.desc().uuid, Y.desc().uuid, kernelH, kernelW, strideH, strideW, padH, padW, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor avgPool2D(cuTensor X, int kernelH, int kernelW, int padH, int padW, int strideH, int strideW){
        unsigned int n = X.desc().sizes.n;
        unsigned int c = X.desc().sizes.c;
        unsigned int h = (X.desc().sizes.h + 2 * padH - kernelH) / strideH + 1;
        unsigned int w = (X.desc().sizes.w + 2 * padW - kernelW) / strideW + 1;
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, n, c, h, w);
        return avgPool2D(X, Y, kernelH, kernelW, padH, padW, strideH, strideW);
    }
    
    cuTensor globalAvgPool2D(cuTensor X, cuTensor Y){
        globalAvgPoolOp(X.impl, Y.impl, 1, 1);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new GLOBAL_AVGPOOL2D(X.desc().uuid, Y.desc().uuid, 1, 1);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor globalAvgPool2D(cuTensor X, cuTensor Y, float alpha1, float alpha2){
        globalAvgPoolOp(X.impl, Y.impl, alpha1, alpha2);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new GLOBAL_AVGPOOL2D(X.desc().uuid, Y.desc().uuid, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor globalAvgPool2D(cuTensor X){
        unsigned int n = X.desc().sizes.n;
        unsigned int c = X.desc().sizes.c;
        unsigned int h = 1;
        unsigned int w = 1;
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, n, c, h, w);
        return globalAvgPool2D(X, Y);
    }
    
    cuTensor flatten(cuTensor X, cuTensor Y){
        flattenOp(X.impl, Y.impl);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new FLATTEN(X.desc().uuid, Y.desc().uuid);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor flatten(cuTensor X){
        unsigned int n = X.desc().sizes.n;
        unsigned int c = X.desc().sizes.c;
        unsigned int h = X.desc().sizes.h;
        unsigned int w = X.desc().sizes.w;
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, n, 1, 1, c * h * w);
        return flatten(X, Y);
    }
    
    //--------------------------------------------------------------------------------
    //Activations
    
    cuTensor relu(cuTensor X){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return relu(X, Y);
    }
    
    cuTensor relu(cuTensor X, cuTensor Y){
        reluOp(X.impl, Y.impl, 1, 0);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new RELU(X.desc().uuid, Y.desc().uuid, 1, 0);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor relu(cuTensor X, cuTensor Y, float alpha1, float alpha2){
        reluOp(X.impl, Y.impl, alpha1, alpha2);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new RELU(X.desc().uuid, Y.desc().uuid, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor sigmoid(cuTensor X){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return sigmoid(X, Y);
    }
    
    cuTensor sigmoid(cuTensor X, cuTensor Y){
        sigmoidOp(X.impl, Y.impl, 1, 0);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new SIGMOID(X.desc().uuid, Y.desc().uuid, 1, 0);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor sigmoid(cuTensor X, cuTensor Y, float alpha1, float alpha2){
        sigmoidOp(X.impl, Y.impl, alpha1, alpha2);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new SIGMOID(X.desc().uuid, Y.desc().uuid, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor tanh(cuTensor X){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return tanh(X, Y);
    }
    
    cuTensor tanh(cuTensor X, cuTensor Y){
        tanhOp(X.impl, Y.impl, 1, 0);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new TANH(X.desc().uuid, Y.desc().uuid, 1, 0);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor tanh(cuTensor X, cuTensor Y, float alpha1, float alpha2){
        tanhOp(X.impl, Y.impl, alpha1, alpha2);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new TANH(X.desc().uuid, Y.desc().uuid, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor elu(cuTensor X, float alpha){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return elu(X, Y, alpha);
    }
    
    cuTensor elu(cuTensor X, cuTensor Y, float alpha){
        eluOp(X.impl, Y.impl, alpha, 1, 0);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new ELU(X.desc().uuid, Y.desc().uuid, alpha, 1, 0);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor elu(cuTensor X, cuTensor Y, float alpha, float alpha1, float alpha2){
        eluOp(X.impl, Y.impl, alpha, alpha1, alpha2);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new ELU(X.desc().uuid, Y.desc().uuid, alpha, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor swish(cuTensor X, float beta){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return swish(X, Y, beta);
    }
    
    cuTensor swish(cuTensor X, cuTensor Y, float beta){
        swishOp(X.impl, Y.impl, beta, 1, 0);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new SWISH(X.desc().uuid, Y.desc().uuid, beta, 1, 0);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor swish(cuTensor X, cuTensor Y, float beta, float alpha1, float alpha2){
        swishOp(X.impl, Y.impl, beta, alpha1, alpha2);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new SWISH(X.desc().uuid, Y.desc().uuid, beta, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor clippedRelu(cuTensor X, float threshold){
        cuTensor Y = cuTensor::create(X.data()->deviceID, X.desc().dType, X.desc().sizes.n,
                                      X.desc().sizes.c, X.desc().sizes.h, X.desc().sizes.w);
        return clippedRelu(X, Y, threshold);
    }
    
    cuTensor clippedRelu(cuTensor X, cuTensor Y, float threshold){
        clippedReluOp(X.impl, Y.impl, threshold, 1, 0);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new CLIPPED_RELU(X.desc().uuid, Y.desc().uuid, threshold, 1, 0);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor clippedRelu(cuTensor X, cuTensor Y, float threshold, float alpha1, float alpha2){
        clippedReluOp(X.impl, Y.impl, threshold, alpha1, alpha2);
        
        if(regisModeCTX){
            //push forward instruction
            auto* inst = new CLIPPED_RELU(X.desc().uuid, Y.desc().uuid, threshold, alpha1, alpha2);
            forwardOpsCTX.push_back(inst);
        }
        return Y;
    }
    
    cuTensor randUniform(cuTensor A, double min, double max){
        return A.randUniform(min, max);
    }
    
    cuTensor randNormal(cuTensor A, double mean, double stddev){
        return A.randNormal(mean, stddev);
    }
}
