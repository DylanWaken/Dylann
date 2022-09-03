#include <iostream>

#include "dylann/tensor/cuTensor.cuh"
#include "dylann/tensor/shell.cuh"

using namespace dylann;

int main() {
    auto X = cuTensor::create<CUDNN_DATA_HALF>(0,3,20).randNormal(1,0);
    auto Y = cuTensor::create<CUDNN_DATA_HALF>(0,1,3);
    cudaMemset(X.impl->data->data, 0, 20*2);
    
    reduceOp(X.impl, Y.impl, 20);
    Y.print();
}
