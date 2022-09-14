//
// Created by Dylan on 9/6/2022.
//

#include "DylannContext.cuh"

namespace dylann {
    
    map<TENSOR_PTR ,cuTensorBase*> tensorsCTX;
    map<TENSOR_PTR ,cuTensorBase*> paramsCTX;   //optimizers will be applied on these
    
    vector<Operation*> forwardOpsCTX;
    vector<Operation*> backwardOpsCTX;
    
    bool regisModeCTX;
    unsigned int tensorIDSeqCTX;
    
    void initEngineContext(){
        cudaMallocHost(&tensorIDSeqG, sizeof(unsigned int));
        *tensorIDSeqG = tensorIDSeqCTX;
        
        cuTensorBase::tensorPoolG = &tensorsCTX;
    }
    
    void beganModelRegister(){
        regisModeCTX = true;
        onModelRegisterG = true;
    }
    
    void endModelRegister(){
        regisModeCTX = false;
        onModelRegisterG = false;
    }
} // dylann