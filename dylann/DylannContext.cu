//
// Created by Dylan on 9/6/2022.
//

#include "DylannContext.cuh"

namespace dylann {
    void initEngineContext(){
        cudaMallocHost(&engineContextG, sizeof(DylannBase));
        *tensorIDSeqG = engineContextG->tensorIDSeq;
        onModelRegisterG = engineContextG->regisMode;
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