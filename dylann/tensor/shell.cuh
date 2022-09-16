//
// Created by Dylan on 8/8/2022.
//

#ifndef DYLANN_SHELL_CUH
#define DYLANN_SHELL_CUH

#include "cuTensor.cuh"
#include "../serial/Instructions.cuh"
#include "../serial/GradInstructions.cuh"
#include "../serial/AutoGrad.cuh"
#include "../DylannContext.cuh"


namespace dylann{
    //these are the "operations" defined for cuTensorBase
    //but implemented with "cuTensor" with gradient tracking functionalities
    cuTensor add(cuTensor& A, cuTensor& B, float alpha, float beta);
    
    cuTensor scale(cuTensor& A, float alpha);
    
    cuTensor linear(cuTensor& W, cuTensor& B, cuTensor& X, cuTensor& Y);
    cuTensor linear(cuTensor& W, cuTensor& B, cuTensor& X);
    cuTensor linear(cuTensor& X, int outDim);
    
    cuTensor conv2D(cuTensor& X, cuTensor& W, cuTensor& B, cuTensor& Y,
                     int padH, int padW, int strideH, int strideW, int dilationH, int dilationW);
    cuTensor conv2D(cuTensor& X, cuTensor& W, cuTensor& B,
                    int padH, int padW, int strideH, int strideW, int dilationH, int dilationW);
    cuTensor conv2D(cuTensor& X, int kernelH, int kernelW, int outChannels,
                    int padH, int padW, int strideH, int strideW, int dilationH, int dilationW);
    
    cuTensor reduce(cuTensor& X, cuTensor& Y, int step);
    
    cuTensor softmax(cuTensor& X, cuTensor& Y, int step);
    cuTensor softmax(cuTensor& X, int step);
    
    cuTensor softmaxLog(cuTensor& X, cuTensor& Y, int step);
    cuTensor softmaxLog(cuTensor& X, int step);
    
    cuTensor softmaxCE(cuTensor& X, cuTensor& Y, int step);
    cuTensor softmaxCE(cuTensor& X, int step);
    
    /**
     * @param Xs: input tensors
     * @param Xs
     * @param Y
     * @param XGradTarget The main tensor that the chained backward computation will continue on
     * @return
     */
    cuTensor channelConcat(std::initializer_list<cuTensor> Xs, cuTensor& Y);
    cuTensor channelConcat(std::initializer_list<cuTensor> Xs);
    
    cuTensor channelConcat(cuTensor* Xs, int inputCount, cuTensor& Y);
    cuTensor channelConcat(cuTensor* Xs, int inputCount);
    
    cuTensor batchnorm(cuTensor& X, cuTensor& Y, cuTensor& runningMean, cuTensor& runningVar,
                       cuTensor& gamma, cuTensor& beta, float eps, float expAvgFactor, float alpha1, float alpha2);
    cuTensor batchnorm(cuTensor& X, cuTensor& Y, cuTensor& runningMean, cuTensor& runningVar,
                       cuTensor& gamma, cuTensor& beta, float eps, float expAvgFactor);
    cuTensor batchnorm(cuTensor& X, cuTensor& runningMean, cuTensor& runningVar,
                         cuTensor& gamma, cuTensor& beta, float eps, float expAvgFactor);
    cuTensor batchnorm(cuTensor& X, float eps, float expAvgFactor);
    
    
    cuTensor dropout(cuTensor& X, cuTensor& Y, float p);
    cuTensor dropout(cuTensor& X, float p);
    
    cuTensor maxPool2D(cuTensor& X, cuTensor& Y, int kernelH, int kernelW, int padH, int padW, int strideH, int strideW);
    cuTensor maxPool2D(cuTensor& X, int kernelH, int kernelW, int padH, int padW, int strideH, int strideW);
    
    cuTensor avgPool2D(cuTensor& X, cuTensor& Y, int kernelH, int kernelW, int padH, int padW, int strideH, int strideW);
    cuTensor avgPool2D(cuTensor& X, int kernelH, int kernelW, int padH, int padW, int strideH, int strideW);
    
    cuTensor globalAvgPool2D(cuTensor& X, cuTensor& Y);
    cuTensor globalAvgPool2D(cuTensor& X);

    cuTensor flatten(cuTensor& X, cuTensor& Y);
    cuTensor flatten(cuTensor& X);
    
    
    //Activations
    cuTensor relu(cuTensor& X);
    cuTensor relu(cuTensor& X, cuTensor& Y);
    cuTensor relu(cuTensor& X, cuTensor& Y, float alpha1, float alpha2);
    
    cuTensor sigmoid(cuTensor& X);
    cuTensor sigmoid(cuTensor& X, cuTensor& Y);
    cuTensor sigmoid(cuTensor& X, cuTensor& Y, float alpha1, float alpha2);
    
    cuTensor tanh(cuTensor& X);
    cuTensor tanh(cuTensor& X, cuTensor& Y);
    cuTensor tanh(cuTensor& X, cuTensor& Y, float alpha1, float alpha2);
    
    cuTensor elu(cuTensor& X, float alpha);
    cuTensor elu(cuTensor& X, cuTensor& Y, float alpha);
    cuTensor elu(cuTensor& X, cuTensor& Y, float alpha, float alpha1, float alpha2);
    
    cuTensor swish(cuTensor& X, float beta);
    cuTensor swish(cuTensor& X, cuTensor& Y, float beta);
    cuTensor swish(cuTensor& X, cuTensor& Y, float beta, float alpha1, float alpha2);
    
    cuTensor clippedRelu(cuTensor& X, float threshold);
    cuTensor clippedRelu(cuTensor& X, cuTensor& Y, float threshold);
    cuTensor clippedRelu(cuTensor& X, cuTensor& Y, float threshold, float alpha1, float alpha2);
    
    //rand init
    cuTensor randUniform(cuTensor& A, double min, double max);
    
    cuTensor randNormal(cuTensor& A, double mean, double stddev);
}


#endif //DYLANN_SHELL_CUH
