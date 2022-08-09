#include "opRegistry.cuh"

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
        
        int a = 1, b = 1;
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
                                  &a,
                                  A->desc.cudnnDesc,
                                  A->grad->data,
                                  &b,
                                  A->desc.cudnnDesc,
                                  A->gradBuf->data
        ));
        return A;
    }
    
    void GRAD_ADD_A::backward(cuTensorBase *current) {
        assert(current->desc.withGrad);
        cudaSetDevice(current->data->deviceID);
    
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                    current->desc.cudnnDesc,
                                    current->grad->data,
                                    &alpha));
    }
    
    void GRAD_ADD_B::backward(cuTensorBase *current) {
        assert(prev->desc.withGrad);
        cudaSetDevice(current->data->deviceID);
    
        if (prev->desc.withGradBuf){
            cudaMemcpy(prev->gradBuf->data, current->grad->data, prev->grad->memSize, cudaMemcpyDeviceToDevice);
            assertCuda(__FILE__, __LINE__);
        
            checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                        prev->desc.cudnnDesc,
                                        prev->gradBuf->data,
                                        &beta))
            mergeGradBuf(prev);
            return;
        }
    
        cudaMemcpy(prev->grad->data, current->grad->data, current->grad->memSize, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
    
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                    prev->desc.cudnnDesc,
                                    prev->grad->data,
                                    &beta))
    }
    
    void GRAD_SCALE::backward(cuTensorBase *current) {
        assert(current->desc.withGrad);
        cudaSetDevice(current->data->deviceID);
    
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                    current->desc.cudnnDesc,
                                    current->grad->data,
                                    &alpha));
    }
}