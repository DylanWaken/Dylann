//
// Created by Dylan on 8/8/2022.
//

#ifndef DYLANN_SHELL_CUH
#define DYLANN_SHELL_CUH

#include "cuTensor.cuh"
#include "../ops/cuLinear.cuh"
#include "../ops/cuConv.cuh"
#include "../ops/cuActivation.cuh"
#include "../ops/cuReduce.cuh"
#include "../ops/cuConcat.cuh"
#include "../ops/cuBatchnorm.cuh"
#include "../ops/cuPool.cuh"


namespace dylann{
    //these are the "operations" defined for cuTensorBase
    //but implemented with "cuTensor" with gradient tracking functionalities
    cuTensor add(cuTensor& A, cuTensor& B, float alpha, float beta);
    
    cuTensor scale(cuTensor& A, float alpha);
    
    cuTensor linear(cuTensor& W, cuTensor& B, cuTensor& X, cuTensor& Y);
    
    cuTensor conv2D(cuTensor& X, cuTensor& W, cuTensor& B, cuTensor& Y,
                     int padH, int padW, int strideH, int strideW, int dilationH, int dilationW);
    
    cuTensor reduce(cuTensor& X, cuTensor& Y, int step);
    
    cuTensor softmax(cuTensor& X, cuTensor& Y, int step);
    cuTensor softmaxLog(cuTensor& X, cuTensor& Y, int step);
    cuTensor softmaxCE(cuTensor& X, cuTensor& Y, int step);
    
    /**
     * @param Xs: input tensors
     * @param Xs
     * @param Y
     * @param XGradTarget The main tensor that the chained backward computation will continue on
     * @return
     */
    cuTensor channelConcat(std::initializer_list<cuTensor> Xs, cuTensor& Y, cuTensor& XGradTarget);
    cuTensor channelConcat(cuTensor* Xs, int inputCount, cuTensor& Y, cuTensor& XGradTarget);
    
    //Activations
    cuTensor relu(cuTensor& X);
    cuTensor relu(cuTensor& X, cuTensor& Y);
    
    cuTensor sigmoid(cuTensor& X);
    cuTensor sigmoid(cuTensor& X, cuTensor& Y);
    
    cuTensor tanh(cuTensor& X);
    cuTensor tanh(cuTensor& X, cuTensor& Y);
    
    cuTensor elu(cuTensor& X, float alpha);
    cuTensor elu(cuTensor& X, cuTensor& Y, float alpha);
    
    cuTensor swish(cuTensor& X, float beta);
    cuTensor swish(cuTensor& X, cuTensor& Y, float beta);
    
    cuTensor clippedRelu(cuTensor& X, float threshold);
    cuTensor clippedRelu(cuTensor& X, cuTensor& Y, float threshold);
    
    //rand init
    cuTensor randUniform(cuTensor& A, double min, double max);
    
    cuTensor randNormal(cuTensor& A, double mean, double stddev);
}


#endif //DYLANN_SHELL_CUH
