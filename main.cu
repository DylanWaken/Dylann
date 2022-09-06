#include <iostream>

#include "dylann/tensor/cuTensor.cuh"
#include "dylann/tensor/shell.cuh"

using namespace dylann;

int main() {
    auto X = cuTensor::create<CUDNN_DATA_FLOAT>(0, 3, 4, 4, 4);
    auto Y = cuTensor::create<CUDNN_DATA_FLOAT>(0, 3, 5, 4, 4).randNormal(1,0);
    auto Z = cuTensor::create<CUDNN_DATA_FLOAT>(0, 3, 9, 4, 4);
    
    cuTensorBase** Xs;
    cudaMallocHost(&Xs, sizeof(cuTensorBase*) * 2);
    Xs[0] = X.impl;
    Xs[1] = Y.impl;
    
    concatChannelOp(Xs, 2, Z.impl);
    
    Z.print();
}
