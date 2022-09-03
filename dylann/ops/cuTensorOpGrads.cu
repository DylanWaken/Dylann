#include "cuTensorOpGrads.cuh"

namespace dylann{
    cuTensorBase* copy(cuTensorBase* A, cuTensorBase*B){
        assertAllocated({A, B});
        cudaSetDevice(A->data->deviceID);
        
        cudaMemcpy(B->data->data, A->data->data, A->data->memSize, cudaMemcpyDeviceToDevice);
        assertCuda(__FILE__, __LINE__);
        return B;
    }
    
    void GRAD_ADD_A::backwardCalc(cuTensorBase *Y) {
        addOpGradA(Y, this->alpha);
    }
    
    void GRAD_ADD_B::backwardCalc(cuTensorBase *Y) {
        addOpGradB(Y, this->target, this->beta);
    }
    
    void GRAD_SCALE::backwardCalc(cuTensorBase *Y) {
        scaleOpGrad(Y, this->alpha);
    }
}