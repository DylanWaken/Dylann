#include <iostream>

#include "dylann/tensor/cuTensor.cuh"

using namespace dylann;

int main() {
     cuTensor W = cuTensor::declare<CUDNN_DATA_FLOAT>(1,1,16, 8).instantiate(0).randNormal(1, 0);
     cuTensor X = cuTensor::declare<CUDNN_DATA_FLOAT>(1,1,2, 16).instantiate(0).randNormal(1, 0);
     cuTensor Y = cuTensor::declare<CUDNN_DATA_FLOAT>(1,1,2,8).instantiate(0).randNormal(0, 0);
     
     linearOp(W.impl, X.impl, Y.impl);
     
     Y.print();
     
}
