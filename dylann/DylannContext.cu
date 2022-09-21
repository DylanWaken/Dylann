//
// Created by Dylan on 9/6/2022.
//

#include "DylannContext.cuh"
#include "tensor/cuTensor.cuh"

namespace dylann {
    
    map<TENSOR_PTR ,cuTensorBase*> tensorsCTX;
    map<TENSOR_PTR ,cuTensorBase*> paramsCTX;   //optimizers will be applied on these
    
    vector<Operation*> forwardOpsCTX;
    vector<Operation*> backwardOpsCTX;
    
    bool regisModeCTX;
    unsigned int tensorIDSeqCTX = 0;
    bool engineAliveCTX = false;
    
    //register mode is automatically started after initialized engine context
    void initEngineContext(){
        *tensorIDSeqG = tensorIDSeqCTX;
        
        cuTensorBase::tensorPoolG = &tensorsCTX;
        cuTensor::instructions = &forwardOpsCTX;
        
        engineAliveCTX = true;
        regisModeCTX = true;
        onModelRegisterG = true;
    }
    
    void beganModelRegister(){
        regisModeCTX = true;
        onModelRegisterG = true;
    }
    
    void allocModelParams(){
        for(auto it : tensorsCTX){
            if(it.second->desc.isParam){
                paramsCTX.insert(it);
            }
        }
    }
    
    void endModelRegister(){
        regisModeCTX = false;
        onModelRegisterG = false;
    }
    
    Sequence* ctx2seq(){
        Sequence* seq;
        cudaMallocHost(&seq, sizeof(Sequence));
        seq->tensorsSeq = tensorsCTX;
        seq->paramsSeq = paramsCTX;
        seq->forwardOpSeq = forwardOpsCTX;
        seq->backwardOpSeq = backwardOpsCTX;
    
        for (auto& op : seq->forwardOpSeq) {
            op->bind(&seq->tensorsSeq);
        }
        
        for (auto& op : seq->backwardOpSeq) {
            op->bind(&seq->tensorsSeq);
        }
    
        tensorsCTX.clear();
        paramsCTX.clear();
        forwardOpsCTX.clear();
        backwardOpsCTX.clear();
    
        *tensorIDSeqG = 0;
        return seq;
    }
    
    Sequence* ctx2SeqExport(){
        Sequence* seq;
        cudaMallocHost(&seq, sizeof(Sequence));
        seq->tensorsSeq = tensorsCTX;
        seq->paramsSeq = paramsCTX;
        seq->forwardOpSeq = forwardOpsCTX;
        seq->backwardOpSeq = backwardOpsCTX;
    
        for (auto& op : seq->forwardOpSeq) {
            op->bind(&seq->tensorsSeq);
        }
    
        for (auto& op : seq->backwardOpSeq) {
            op->bind(&seq->tensorsSeq);
        }
        
        return seq;
    }
} // dylann