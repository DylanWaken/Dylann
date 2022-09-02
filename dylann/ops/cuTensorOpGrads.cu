#include "cuTensorOpGrads.cuh"

namespace dylann{
    cuTensorBase* copy(cuTensorBase* A, cuTensorBase*B){
        assertAllocated({A, B});
        cudaSetDevice(A->data->deviceID);
        
        cudaMemcpy(B->data->data, A->data->data, A->data->memSize, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        return B;
    }
    
    cuTensorBase* mergeGradBuf(cuTensorBase* A){
        assertAllocated({A});
        assert(A->desc.withGradBuf);
        cudaSetDevice(A->data->deviceID);
        
        float a = 1, b = 1;
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
                                  &b,
                                  A->desc.cudnnDesc,
                                  A->gradBuf->data,
                                  &a,
                                  A->desc.cudnnDesc,
                                  A->grad->data
        ))
        return A;
    }
    
    void GRAD_ADD_A::backwardCalc(cuTensorBase *current) {
        assert(current->desc.withGrad);
        cudaSetDevice(current->data->deviceID);
    
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                    current->desc.cudnnDesc,
                                    current->grad->data,
                                    &alpha));
    }
    
    void GRAD_ADD_B::backwardCalc(cuTensorBase *current) {
        assert(target->desc.withGrad);
        cudaSetDevice(current->data->deviceID);
    
        if (target->desc.withGradBuf){
            cudaMemcpy(target->gradBuf->data, current->grad->data, target->grad->memSize, cudaMemcpyDeviceToDevice);
            assertCuda(__FILE__, __LINE__);
        
            checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                        target->desc.cudnnDesc,
                                        target->gradBuf->data,
                                        &beta))
            mergeGradBuf(target);
            return;
        }
    
        cudaMemcpy(target->grad->data, current->grad->data, current->grad->memSize, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
    
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                    target->desc.cudnnDesc,
                                    target->grad->data,
                                    &beta))
    }
    
    void GRAD_SCALE::backwardCalc(cuTensorBase *current) {
        assert(current->desc.withGrad);
        cudaSetDevice(current->data->deviceID);
    
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                    current->desc.cudnnDesc,
                                    current->grad->data,
                                    &alpha));
    }
}