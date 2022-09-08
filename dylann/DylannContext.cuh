//
// Created by Dylan on 9/6/2022.
//

#ifndef DYLANN_DYLANNCONTEXT_CUH
#define DYLANN_DYLANNCONTEXT_CUH

#include <vector>
#include "serial/Instructions.cuh"

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
    class DylannBase {
    public:
        vector<cuTensorBase*> tensors;
        vector<cuTensorBase*> params;   //optimizers will be applied on these
        
        vector<Operation> forwardOps;
        vector<Operation> backwardOps;
        
        bool regisMode;
        unsigned int tensorIDSeq = 0;
    };
    
    extern DylannBase* engineContextG;
    
    void initEngineContext();
    
    void beganModelRegister();
    
    void endModelRegister();
    
} // dylann

#endif //DYLANN_DYLANNCONTEXT_CUH
