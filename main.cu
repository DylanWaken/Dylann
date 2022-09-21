#include <iostream>

#include "presets/Resnet.cuh"
#include "dylann/optim/Optimizers.cuh"

using namespace dylann;

int main() {
    initEngineContext();
    beganModelRegister();
    
    auto X = cuTensor::create<CUDNN_DATA_FLOAT>(shape4(64, 64, 32, 32), 0);
    
    ResnetConv id = ResnetConv();
    auto Y = id.forward(X);
    auto Z = id.forward(Y);
    auto W = id.forward(Z);
    
    generateGrads(forwardOpsCTX, backwardOpsCTX);
    allocModelParams();
    
    auto opt = Adam(0.001);
    opt.bindDefaultParams();
}
