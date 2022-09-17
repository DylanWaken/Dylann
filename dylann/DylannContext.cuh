//
// Created by Dylan on 9/6/2022.
//

#ifndef DYLANN_DYLANNCONTEXT_CUH
#define DYLANN_DYLANNCONTEXT_CUH

#include <vector>
#include "serial/Instructions.cuh"
#include "serial/GradInstructions.cuh"
#include "tensor/cuTensor.cuh"

using namespace std;
namespace dylann {
    
    /**
     * VERY IMPORTANT
     *
     * This is the handle the entire engine would depend on once the model starts running
     * It keep track of all tensors UUIDs
     * All model parameters
     * The forward and backward instructions
     *
     */
    extern map<TENSOR_PTR ,cuTensorBase*> tensorsCTX;
    extern map<TENSOR_PTR ,cuTensorBase*> paramsCTX;   //optimizers will be applied on these
    
    extern vector<Operation*> forwardOpsCTX;
    extern vector<Operation*> backwardOpsCTX;
    
    extern bool regisModeCTX;
    extern unsigned int tensorIDSeqCTX;
    
    void initEngineContext();
    
    void beganModelRegister();
    
    void allocModelParams();
    
    void endModelRegister();
    
} // dylann

#endif //DYLANN_DYLANNCONTEXT_CUH
