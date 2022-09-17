#include <iostream>

#include "dylann/tensor/shell.cuh"
#include "dylann/optim/Optimizers.cuh"

using namespace dylann;

int main() {
    initEngineContext();
    beganModelRegister();
    
    auto X = cuTensor::create<CUDNN_DATA_FLOAT>(shape4(64, 3, 32, 32), 0);
    
    X = conv2D(X, 3, 3, 64, 1, 1, 1, 1, 1, 1);
    X = relu(X);
    X = maxPool2D(X, 2, 2, 0, 0, 2, 2);
    
    X = conv2D(X, 3, 3, 64, 1, 1, 1, 1, 1, 1);
    X = relu(X);
    X = maxPool2D(X, 2, 2, 0, 0, 2, 2);
    
    X = conv2D(X, 3, 3, 128, 1, 1, 1, 1, 1, 1);
    X = relu(X);
    X = maxPool2D(X, 2, 2, 0, 0, 2, 2);
    
    X = flatten(X);
    X = linear(X, 10);
    X = softmaxCE(X, X.sizes().size / X.sizes().n);
    
    for (auto & i : forwardOpsCTX) {
        i->bind(&tensorsCTX);
        i->run();
    }
    
    generateGrads(forwardOpsCTX, backwardOpsCTX);
    allocModelParams();
    
    auto opt = Adam(0.001);
    opt.bindDefaultParams();
    
    for (int p = 0; p < 100000; p++) {
        for (auto &i: backwardOpsCTX) {
            i->run();
            opt.apply();
        }
    }
    
}
