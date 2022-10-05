//
// Created by Dylan on 10/4/2022.
//
#include <iostream>

#include "presets/Resnet.cuh"
#include "dylann/module/Sequence.cuh"

#include <thread>
#include <mutex>
#include <condition_variable>
#include "io/dataset/Dataset.cuh"
#include "presets/readFuncs/BuildinReadfuncs.cuh"
#include <fstream>
#define MINI_BATCH_SIZE 4

using namespace dylann;
using namespace std;
using namespace io;


int main() {
    initEngineContext();
    auto X0 = cuTensor::create(0, CUDNN_DATA_FLOAT, {MINI_BATCH_SIZE, 1,1,64}).randNormal(0, 1);
    auto X1 = linear(X0, 32);
    X1 = relu(X1);
    auto X2 = linear(X1, 10);
    auto Y = softmaxCE(X2, 10);
    
    auto X0Buf = cuTensor::create(0, CUDNN_DATA_FLOAT, {MINI_BATCH_SIZE, 1,1,64});
    auto label = cuTensor::create(0, CUDNN_DATA_FLOAT, {MINI_BATCH_SIZE, 1, 1, 10});
    
    float answer[MINI_BATCH_SIZE * 10] = {
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    };
    
    cudaMemcpy(label.dataPtr(), answer, sizeof(float) * MINI_BATCH_SIZE * 10, cudaMemcpyHostToDevice);
    
    tensorsCTX.erase(0x09);
    tensorsCTX.erase(0x0a);
    
    auto seq = ctx2seq();
    seq->generateGrad();
    seq->randomizeParams();
    seq->setOpt(new Momentum(0.1, 0.9));
    seq->setLoss(new CrossEntropy(Y.impl));
    
    for (auto &i : seq->forwardOpSeq) {
        i->print();
    }
    
    for (auto &i : seq->backwardOpSeq) {
        i->print();
    }
    
    X0.randNormal(0, 10);
    
    cudaMemcpy(X0Buf.dataPtr(), X0.dataPtr(), sizeof(float) * MINI_BATCH_SIZE * 64, cudaMemcpyDeviceToDevice);
    seq->forward();
    seq->backward(label.impl);
    
    for (auto &i : seq->tensorsSeq) {
        cuTensor::inherit(i.second).toFile(R"(D:\Projects\PyCharm\DylannValidation\dataBuf\)");
    }
    
    seq->opt->apply();
    seq->resetGrad();
    
    for (int i = 0; i < 0; i++){
        cudaMemcpy(X0.dataPtr(), X0Buf.dataPtr(), sizeof(float) * MINI_BATCH_SIZE * 64, cudaMemcpyDeviceToDevice);
        seq->forward();
        seq->backward(label.impl);
        seq->opt->apply();
        
        seq->resetGrad();
    }
    
    cudaMemcpy(X0.dataPtr(), X0Buf.dataPtr(), sizeof(float) * MINI_BATCH_SIZE * 64, cudaMemcpyDeviceToDevice);
    seq->forward();
    seq->backward(label.impl);
    
    for (auto &i : seq->tensorsSeq) {
        cuTensor::inherit(i.second).toFile(R"(D:\Projects\PyCharm\DylannValidation\dataBuf\opt\)");
    }
    label.toFile(R"(D:\Projects\PyCharm\DylannValidation\dataBuf\)");
}