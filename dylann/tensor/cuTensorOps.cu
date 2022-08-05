//
// Created by Dylan on 8/5/2022.
//

#include "cuTensorOps.cuh"

namespace dylann{
    cuTensor copy(const cuTensor &A, const cuTensor &B){
        assertAllocated({A, B});
        cudaSetDevice(A.getDev());
        
        cudaMemcpy(B.dataPtr(), A.dataPtr(), A.memSize(), cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        return B;
    }
    
    cuTensor add(const cuTensor &A, const cuTensor &B, int alpha, int beta){
        assertAllocated({A, B});
        assertOnSameDev({A, B});
        cudaSetDevice(A.getDev());
        
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
                       &alpha,
                       A.cudnnDesc(),
                       A.dataPtr(),
                       &beta,
                       B.cudnnDesc(),
                       B.dataPtr()))
        return A;
    }
    
    cuTensor add(const cuTensor &A, const cuTensor &B, const cuTensor &C, int alpha, int beta){
        assertAllocated({A, B});
        assertOnSameDev({A, B});
        cudaSetDevice(A.getDev());
        
        copy(A, C);
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
                       &alpha,
                       C.cudnnDesc(),
                       C.dataPtr(),
                       &beta,
                       B.cudnnDesc(),
                       B.dataPtr()))
        return C;
    }
}