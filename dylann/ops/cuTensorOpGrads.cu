#include "cuTensorOpGrads.cuh"

namespace dylann{
    cuTensorBase* copy(cuTensorBase* A, cuTensorBase*B){
        assertAllocated({A, B});
        cudaSetDevice(A->data->deviceID);
        
        cudaMemcpy(B->data->data, A->data->data, A->data->memSize, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        return B;
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
    
        float a = 1.0f;
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
                                  &beta,
                                  current->desc.cudnnDesc,
                                  current->grad->data,
                                  &a,
                                  target->desc.cudnnDesc,
                                  target->grad->data))
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