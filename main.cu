#include <iostream>

#include "presets/Resnet.cuh"
#include "dylann/module/Sequence.cuh"

#include <thread>
#include <mutex>
#include <condition_variable>
#include "io/dataset/Dataset.cuh"
#include "presets/readFuncs/BuildinReadfuncs.cuh"
#define MINI_BATCH_SIZE 64

using namespace dylann;
using namespace std;
using namespace io;

int main() {


    //register model
    initEngineContext();
    auto X0 = cuTensor::create(0, CUDNN_DATA_FLOAT, {MINI_BATCH_SIZE, 3, 32, 32});
    
    ResnetIdentity id = ResnetIdentity();
    ResnetConv cv = ResnetConv();
    
    auto X = conv2D(X0, 3, 3, 64, 1, 1, 1, 1, 1, 1);
    X = batchnorm2d(X, 1e-8, 1);
    X = relu(X);

    for(auto i = 0; i < 5; i++) X = id.forward(X);
    X = cv.forward(X);

    for(auto i = 0; i < 5; i++) X = id.forward(X);
    X = cv.forward(X);

    for(auto i = 0; i < 6; i++) X = id.forward(X);

    auto X2 = flatten(X);
    X2 = linear(X2, 1024);
    X2 = relu(X2);
    X2 = linear(X2, 256);
    X2 = relu(X2);
    auto X3 = linear(X2, 10);
    auto Y = softmaxCE(X3, 10);

    auto seq = ctx2seq();
    seq->generateGrad();
    seq->setLoss(new CrossEntropy(Y.impl));
    seq->setOpt(new Momentum(0.01/64));
    seq->randomizeParams();

    for(auto& i : seq->forwardOpSeq){
        i->print();
    }

    for(auto& i : seq->backwardOpSeq){
        i->print();
    }


    auto* dataset = new DatasetCV(50000, 6400, MINI_BATCH_SIZE, 16,6400,
                                       {1, 3, 32, 32},
                                       shape4(10), CUDNN_DATA_FLOAT);

    ReadFuncCV* readFunc = new CIFAR_10ReadFunc(R"(D:\Resources\Datasets\cifar-10-bin)", 16);
    dataset->bindReadFunc(readFunc);
    dataset->bindAugCV({
        new RandFlip(),
        new RandPadCorp(4),
    });
    dataset->bindAugTensor({new UniformNorm(0, 1)});
    dataset->construct();

    auto label = cuTensor::create(0, CUDNN_DATA_FLOAT, {MINI_BATCH_SIZE, 1, 1,10});
    
    float runningLoss = 0;
    for(int i = 0; i < 500000; i++){
        dataset->nextMiniBatch(X0.impl, label.impl);
        seq->forward();
        
        float loss = seq->getLoss(label.impl);
        runningLoss += loss;
        seq->backward(label.impl);
        seq->opt->apply();
        
        seq->resetGrad();
        
        if(i % 100 == 0 && i != 0){
            cout << runningLoss / 100 << ", ";
            runningLoss = 0;
            
            float valLoss = 0;
            for(int j = 0; j < 50; j++){
                dataset->nextValBatch(X0.impl, label.impl);
                seq->forward();
                valLoss += seq->getLoss(label.impl);
                seq->resetGrad();
            }
            cout << valLoss / 50 << ", ";
        }
    }
}
