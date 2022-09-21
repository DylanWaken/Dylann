//
// Created by Dylan on 9/21/2022.
//

#ifndef DYLANN_SEQUENCE_CUH
#define DYLANN_SEQUENCE_CUH

#include <vector>
#include "../serial/Instructions.cuh"
#include "../serial/GradInstructions.cuh"
#include "../optim/Optimizers.cuh"

namespace dylann{
    class Sequence{
    public:
        map<TENSOR_PTR ,cuTensorBase*> tensorsSeq;
        map<TENSOR_PTR ,cuTensorBase*> paramsSeq;   //optimizers will be applied on these
    
        vector<Operation*> forwardOpSeq;
        vector<Operation*> backwardOpSeq;
        
        OPTIM_BASE* opt = nullptr;
        
        void run(){
            for(auto op : forwardOpSeq){
                op->run();
            }
        }
        
        void generateGrad(){
            generateGrads(forwardOpSeq, backwardOpSeq);
        }
        
        void setOpt(OPTIM_BASE* optm){
             this->opt = optm;
             opt->bindParams(&paramsSeq);
        }
    };
}

#endif //DYLANN_SEQUENCE_CUH
