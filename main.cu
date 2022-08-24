#include <iostream>

#include "dylann/tensor/cuTensor.cuh"

using namespace dylann;

template<typename T>
__global__ void testKernel(T* data, int size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= size) return;
    data[idx] = idx;
}

int main() {
     cuTensor A = cuTensor::declare<CUDNN_DATA_FLOAT>(2, 2, 2, 2).instantiate(0);
     A.randNormal(0, 1);
     A.print();
     
//    testKernel<float><<<1, 16>>>((float*)A.dataPtr(), 16);
//    cudaDeviceSynchronize();
//    assertCuda(__FILE__, __LINE__);
//
//    A.print();
}
