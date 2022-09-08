//
// Created by Dylan on 9/6/2022.
//

#include "DylannContext.cuh"

namespace dylann {
    
    DylannBase* engineContextG;
    
    void initEngineContext(){
        cudaMallocHost(&engineContextG, sizeof(DylannBase));
        *tensorIDSeqG = engineContextG->tensorIDSeq;
        onModelRegisterG = engineContextG->regisMode;
    
        cuTensorBase::tensorPoolG = &engineContextG->tensors;
    }
    
    void beganModelRegister(){
        engineContextG->regisMode = true;
        onModelRegisterG = true;
    }
    
    void endModelRegister(){
        engineContextG->regisMode = false;
        onModelRegisterG = false;
    }
} // dylann