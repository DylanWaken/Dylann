//
// Created by Dylan on 10/1/2022.
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
    auto X0 = cuTensor::create(0, CUDNN_DATA_FLOAT, {MINI_BATCH_SIZE, 1,1,16 * 64}).randNormal(0, 1);
    auto Y = linear(X0, 256);
    Y = relu(Y);
    Y = linear(Y, 10);
    Y = softmax(Y, 10);

    auto label = cuTensor::create(0, CUDNN_DATA_FLOAT, {MINI_BATCH_SIZE, 1, 1, 10});

    auto seq = ctx2seq();
    seq->generateGrad();
    seq->randomizeParams();
    seq->setLoss(new CrossEntropy());
    seq->bindInOut(X0.impl, Y.impl);
    seq->allocModelParams();
    seq->setOpt(new SGD(0.01));

    for (auto &i : seq->forwardOpSeq) {
        i->print();
    }

    for (auto &i : seq->backwardOpSeq) {
        i->print();
    }

    float answer[MINI_BATCH_SIZE * 10] = {
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    };

    cudaMemcpy(label.dataPtr(), answer, sizeof(float) * MINI_BATCH_SIZE * 10, cudaMemcpyHostToDevice);
    seq->forward();
    seq->backward(label.impl);

    for (auto &i : seq->tensorsSeq) {
        cuTensor::inherit(i.second).toFile(R"(D:\Projects\PyCharm\DylannValidation\dataBuf\)");
    }

    for (auto i = 0; i < 10; i++) {
        seq->resetGrad();
        seq->forward();
        cudaMemcpy(label.dataPtr(), answer, sizeof(float) * MINI_BATCH_SIZE * 10, cudaMemcpyHostToDevice);
        seq->backward(label.impl);
        cuTensor::inherit(seq->tensorsSeq[0x6]).print();
        seq->opt->apply();
    }
    

    for (auto &i : seq->tensorsSeq) {
        cuTensor::inherit(i.second).toFile(R"(D:\Projects\PyCharm\DylannValidation\dataBuf\opt\)");
    }

//    cuTensor X = cuTensor::create(0, CUDNN_DATA_FLOAT, {32,1,1}).randNormal(5,0);
//    cuTensor Y = cuTensor::create(0, CUDNN_DATA_FLOAT, {32,1,1}).randNormal(1,0);
//
//    float a = 1, b = -1;
//    cudnnAddTensor(cudnnHdlG,
//                   &b,
//                   Y.impl->desc.cudnnDesc,
//                   Y.impl->data->data,
//                   &a,
//                   X.impl->desc.cudnnDesc,
//                   X.impl->data->data);
//
//    X.print();
}