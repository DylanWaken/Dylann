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
    
    void GRAD_ADD_A::backward(cuTensorBase *current) {
        assert(current->desc.withGrad);
        cudaSetDevice(current->data->deviceID);
    
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                    current->desc.cudnnDesc,
                                    current->grad->data,
                                    &alpha));
    }
    
    void GRAD_ADD_B::backward(cuTensorBase *current) {
        assert(gradSrc->desc.withGrad);
        cudaSetDevice(current->data->deviceID);
    
        if (gradSrc->desc.withGradBuf){
            cudaMemcpy(gradSrc->gradBuf->data, current->grad->data, gradSrc->grad->memSize, cudaMemcpyDeviceToDevice);
            assertCuda(__FILE__, __LINE__);
        
            checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                        gradSrc->desc.cudnnDesc,
                                        gradSrc->gradBuf->data,
                                        &beta))
            mergeGradBuf(gradSrc);
            return;
        }
    
        cudaMemcpy(gradSrc->grad->data, current->grad->data, current->grad->memSize, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
    
        checkCUDNN(cudnnScaleTensor(cudnnHdlG,
                                    gradSrc->desc.cudnnDesc,
                                    gradSrc->grad->data,
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