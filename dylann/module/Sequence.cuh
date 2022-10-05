//
// Created by Dylan on 9/21/2022.
//

#ifndef DYLANN_SEQUENCE_CUH
#define DYLANN_SEQUENCE_CUH

#include <vector>
#include "../serial/Instructions.cuh"
#include "../serial/GradInstructions.cuh"
#include "../optim/Optimizers.cuh"
#include "../optim/Loss.cuh"
#include <fstream>

namespace dylann{
    
    extern unsigned int MAGIC_NUMBER;
    
    class Sequence{
    public:
        map<TENSOR_PTR ,cuTensorBase*> tensorsSeq = {};
        map<TENSOR_PTR ,cuTensorBase*> paramsSeq = {};   //optimizers will be applied on these
        
        cuTensorBase* netX;
        cuTensorBase* netY;
    
        vector<Operation*> forwardOpSeq;
        vector<Operation*> backwardOpSeq;
        
        OPTIM_BASE* opt = nullptr;
        Loss* loss = nullptr;
        
        void forward(){
            for(auto op : forwardOpSeq){
                op->run();
            }
        }
        
        void backward(){
            if (backwardOpSeq.empty()){
                generateGrad();
            }
            
            for(auto op : backwardOpSeq){
                op->run();
            }
        }
        
        void backward(cuTensorBase* target){
            if (backwardOpSeq.empty()){
                generateGrad();
            }
            
            loss->backward(target);
            
            for(auto op : backwardOpSeq){
                op->run();
            }
        }
        
        void resetGrad(){
            for(auto it : tensorsSeq){
                if (!it.second->desc.isParam){
                    it.second->zeroData();
                }
                it.second->zeroGrad();
            }
        }
        
        float getLoss(cuTensorBase* target) const{
            return loss->loss(target);
        }
        
        void allocModelParams(){
            if (paramsSeq.empty()) {
                for (auto it: tensorsSeq) {
                    if (it.second->desc.isParam) {
                        paramsSeq.insert(it);
                    }
                }
            }
        }
        
        void generateGrad(){
            generateGrads(forwardOpSeq, backwardOpSeq);
        }
        
        void setOpt(OPTIM_BASE* optm){
            allocModelParams();
            this->opt = optm;
            opt->bindParams(&paramsSeq);
        }
        
        void setLoss(Loss* modelLoss){
            this->loss = modelLoss;
        }
        
        void randomizeParams(){
            allocModelParams();
            for(auto it : paramsSeq){
                initParamOp(it.second);
            }
        }
        
        void bindInputs(cuTensorBase* X){
            assert(tensorsSeq.find(X->desc.uuid) != tensorsSeq.end()
            && tensorsSeq[X->desc.uuid] == X);
            this->netX = X;
        }
        
        void bindOutputs(cuTensorBase* Y){
            assert(tensorsSeq.find(Y->desc.uuid) != tensorsSeq.end()
            && tensorsSeq[Y->desc.uuid] == Y);
            this->netY = Y;
        }
        
        void bindInOut(cuTensorBase* X, cuTensorBase* Y){
            assert(tensorsSeq.find(X->desc.uuid) != tensorsSeq.end()
                   && tensorsSeq[X->desc.uuid] == X);
            assert(tensorsSeq.find(Y->desc.uuid) != tensorsSeq.end()
                   && tensorsSeq[Y->desc.uuid] == Y);
            this->netX = X;
            this->netY = Y;
        }
        
        void toFile(const string& basePath, const string& saveName);
    };
}

#endif //DYLANN_SEQUENCE_CUH
