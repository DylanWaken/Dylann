#include <iostream>

#include "dylann/tensor/cuTensor.cuh"
#include "dylann/tensor/shell.cuh"

using namespace dylann;

int main() {
     cuTensor W = cuTensor::declare<CUDNN_DATA_FLOAT>(1,1,12, 6).instantiate(0).randNormal(1, 2);
     cuTensor X = cuTensor::declare<CUDNN_DATA_FLOAT>(1,1,8, 12).instantiate(0).randNormal(1, 2);
     cuTensor Y = cuTensor::declare<CUDNN_DATA_FLOAT>(1,1,8,6).instantiate(0).randNormal(0, 0);
     cuTensor B = cuTensor::declare<CUDNN_DATA_FLOAT>(1,1,1,6).instantiate(0).randNormal(0, 0);
     
     linear(W, B, X, Y);
     randNormalGradOp(Y.impl, 1, 0);
     Y.backward();
    
     Y.print();
     W.print();
}
