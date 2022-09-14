#include <iostream>

#include "dylann/tensor/shell.cuh"

using namespace dylann;

int main() {
    initEngineContext();
    beganModelRegister();
    
//    auto X = cuTensor::create<CUDNN_DATA_FLOAT>(0, 2,3,32,32);
//    X = conv2D(X,3,3,6, 1,1, 1, 1,1,1);
//    X = conv2D(X,3,3,6, 1,1, 1, 1,1,1);
//    X = conv2D(X,3,3,6, 1,1, 1, 1,1,1);
//    X = conv2D(X,3,3,6, 1,1, 1, 1,1,1);
//    auto Y = flatten(X);
//    auto Z = linear(Y, 10);

    auto W = cuTensor::create<CUDNN_DATA_FLOAT>(0,4,6).randNormal(1,0);
    auto X = cuTensor::create<CUDNN_DATA_FLOAT>(0,2,1,1,6).randNormal(1,0);
    auto Y = cuTensor::create<CUDNN_DATA_FLOAT>(0,2,1,1,4);
    auto B = cuTensor::create<CUDNN_DATA_FLOAT>(0,1,4).randNormal(1,0);
    
    randNormalGradOp(Y.impl, 1, 0);
    linearOpGrads(W.impl, B.impl, X.impl, Y.impl);
    
    for(auto p : forwardOpsCTX){
        p->print();
    }
    //Y.print();
    W.print();
}
