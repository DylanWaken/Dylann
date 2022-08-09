//
// Created by Dylan on 8/5/2022.
//

#include "cuTensorOps.cuh"

namespace dylann{
    
    cuTensorBase* add(cuTensorBase* A, cuTensorBase* B, float alpha, float beta){
        assertAllocated({A, B});
        assertOnSameDev({A, B});
        cudaSetDevice(A->data->deviceID);
        
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
                       &alpha,
                       A->desc.cudnnDesc,
                       A->data->data,
                       &beta,
                       B->desc.cudnnDesc,
                       B->data->data))
        return A;
    }
    
    cuTensorBase* scale(cuTensorBase* A, float alpha){
        assertAllocated({A});
        cudaSetDevice(A->data->deviceID);
        
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                       A->desc.cudnnDesc,
                       A->data->data,
                       &alpha))
        return A;
    }
}