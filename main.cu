#include <iostream>

#include "dylann/tensor/cuTensor.cuh"
#include "dylann/tensor/shell.cuh"

using namespace dylann;

int main() {
    auto X = cuTensor::create<CUDNN_DATA_FLOAT>(0, 3,32,32).randNormal(1,0);
    auto W = cuTensor::create<CUDNN_DATA_FLOAT>(0, 3,3,3,3).randNormal(1,0);
    auto B = cuTensor::create<CUDNN_DATA_FLOAT>(0, 3,1,1).randNormal(1,0);
    auto Y = cuTensor::create<CUDNN_DATA_FLOAT>(0, 3,32,32);
    
    conv2D(X, W, B, Y, 1,1,1,1,1,1);
    Y += X;
    randNormalGradOp(Y.impl, 1, 0);
    Y.backward();
    
    W.print();
}
