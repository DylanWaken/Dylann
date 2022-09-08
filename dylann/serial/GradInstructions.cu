//
// Created by Dylan on 9/7/2022.
//

#include "GradInstructions.cuh"

namespace dylann {
    void ADD_GRADS::run() {
        addOpGrad((*params)[A], (*params)[B], alpha, beta);
    }
    
    void ADD_GRADS::encodeParams(unsigned char * file, size_t &offset){
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        memcpy(file + offset, &A, sizeof(TENSOR_PTR));
        offset += sizeof(TENSOR_PTR);
        memcpy(file + offset, &B, sizeof(TENSOR_PTR));
        offset += sizeof(TENSOR_PTR);
        memcpy(file + offset, &alpha, sizeof(float));
        offset += sizeof(float);
        memcpy(file + offset, &beta, sizeof(float));
        offset += sizeof(float);
    }
    
    void SCALE_GRADS::run() {
    
    }
} // dylann