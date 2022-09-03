#include <iostream>

#include "dylann/tensor/cuTensor.cuh"
#include "dylann/tensor/shell.cuh"

using namespace dylann;

int main() {
    auto X = cuTensor::create<CUDNN_DATA_FLOAT>(0,3,2).randNormal(1,0);
    auto Y = cuTensor::create<CUDNN_DATA_FLOAT>(0,3,2);
    randNormalGradOp(Y.impl, 0.5,0);
    
    float grads[] = { 1, 0 , 0, 1, 1, 0};
    cudaMemcpy(Y.gradPtr(), grads, sizeof(float) * 6, cudaMemcpyHostToDevice);
    
    softmaxOp(X.impl, Y.impl, 2);
    softmaxOpGrads(X.impl, Y.impl, 2);
    
    Y.print();
    X.print();
}
