//
// Created by Dylan on 9/16/2022.
//

#include "Optimizers.cuh"

void dylann::SGD::apply() {
    float b = - LEARNING_RATE;
    float alphaW = 1 - L2, alphaB = 1;
    for (auto p : (*paramsRes)) {
        
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
               &b, p.second->desc.cudnnDesc, p.second->grad->data,
               p.second->desc.isWeight? &alphaW : &alphaB, p.second->desc.cudnnDesc, p.second->data->data
        ))
        cudaMemset(p.second->grad->data, 0, p.second->data->memSize);
        assertCuda(__FILE__, __LINE__);
    }
}

void dylann::SGD::zeroGrad() {

}

void dylann::Momentum::bindParams(map<TENSOR_PTR, dylann::cuTensorBase *> *params) {
    OPTIM_BASE::bindParams(params);
    for(auto& param : *params){
        optimBufCTX.insert({param.first, cuTensorBase::create(param.second->desc.sizes, param.second->desc.dType)
                ->instantiate(param.second->data->deviceID)});
    }
}

void dylann::Momentum::bindDefaultParams() {
    OPTIM_BASE::bindDefaultParams();
    for(auto& param : *paramsRes){
        optimBufCTX.insert({param.first, cuTensorBase::create(param.second->desc.sizes, param.second->desc.dType)
                ->instantiate(param.second->data->deviceID)});
    }
}

//Momentum : m[t] = m[t-1] * β + (1 - β) * g[t]
//           w[t] = w[t-1] - η * m[t]
void dylann::Momentum::apply() {
    float m1 = BETA, m2 = 1 - BETA, aW = 1 - L2, aB = 1, b = - LEARNING_RATE;
    for (auto p : (*paramsRes)) {
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
               &m2, p.second->desc.cudnnDesc, p.second->grad->data,
               &m1, optimBufCTX[p.first]->desc.cudnnDesc, optimBufCTX[p.first]->data->data
        ))
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
               &b, optimBufCTX[p.first]->desc.cudnnDesc, optimBufCTX[p.first]->data->data,
               p.second->desc.isWeight? &aW : &aB, p.second->desc.cudnnDesc, p.second->data->data
        ))
    
        cudaMemset(p.second->grad->data, 0, p.second->data->memSize);
        assertCuda(__FILE__, __LINE__);
    }
}

void dylann::Momentum::zeroGrad() {
    for (auto p : optimBufCTX) {
        cudaMemset(p.second->data->data, 0, p.second->data->memSize);
        assertCuda(__FILE__, __LINE__);
    }
}

void dylann::RMSProp::bindParams(map<TENSOR_PTR, dylann::cuTensorBase *> *params) {
    OPTIM_BASE::bindParams(params);
    for(auto& param : *params){
        optimBufCTX.insert({param.first, cuTensorBase::create(param.second->desc.sizes, param.second->desc.dType)
                ->instantiate(param.second->data->deviceID)});
    }
}

void dylann::RMSProp::bindDefaultParams() {
    OPTIM_BASE::bindDefaultParams();
    for(auto& param : *paramsRes){
        optimBufCTX.insert({param.first, cuTensorBase::create(param.second->desc.sizes, param.second->desc.dType)
                ->instantiate(param.second->data->deviceID)});
    }
}

//RMSProp : V[t] = V[t-1] * β + (1 - β) * g[t]^2
//          w[t] = w[t-1] - η * g[t] / sqrt(V[t] + ε)

//the backward in optimBufCTX is actually V[t]
void dylann::RMSProp::apply() {
    for (auto p : (*paramsRes)) {
        RSMPropV(p.second->desc.dType, optimBufCTX[p.first]->grad->data,
                 p.second->grad->data, BETA, p.second->desc.numel);
        RSMPropA(p.second->desc.dType, p.second->data->data, optimBufCTX[p.first]->grad->data,
                 p.second->grad->data, EPSILON, LEARNING_RATE, L2, p.second->desc.numel);
        
        cudaMemset(p.second->grad->data, 0, p.second->data->memSize);
        assertCuda(__FILE__, __LINE__);
    }
}

void dylann::RMSProp::zeroGrad() {
    for (auto p : optimBufCTX) {
        cudaMemset(p.second->grad->data, 0, p.second->data->memSize);
        assertCuda(__FILE__, __LINE__);
    }
}

void dylann::Adam::bindParams(map<TENSOR_PTR, dylann::cuTensorBase *> *params) {
    OPTIM_BASE::bindParams(params);
    for(auto& param : *params){
        optimBufCTX.insert({param.first, cuTensorBase::create(param.second->desc.sizes, param.second->desc.dType)
                ->instantiate(param.second->data->deviceID)});
    }
}

void dylann::Adam::bindDefaultParams() {
    OPTIM_BASE::bindDefaultParams();
    for(auto& param : *paramsRes){
        optimBufCTX.insert({param.first, cuTensorBase::create(param.second->desc.sizes, param.second->desc.dType)
                ->instantiate(param.second->data->deviceID)});
    }
}

void dylann::Adam::apply() {
    for (auto p : (*paramsRes)) {
        
        //first order momentum
        float m1 = BETA1, m2 = 1-BETA1;
        checkCUDNN(cudnnAddTensor(cudnnHdlG,
                                  &m2, p.second->desc.cudnnDesc, p.second->grad->data,
                                  &m1, optimBufCTX[p.first]->desc.cudnnDesc, optimBufCTX[p.first]->data->data
        ))
        
        //second order momentum
        RSMPropV(p.second->desc.dType, optimBufCTX[p.first]->grad->data,
                 p.second->grad->data, BETA2, p.second->desc.numel);
        
        //iteraion counter
        float val1 = 1.0f / (1.0f - (float)pow(BETA1, t+1));
        float val2 = 1.0f / (1.0f - (float)pow(BETA2, t+1));
        checkCUDNN(cudnnScaleTensor(cudnnHdlG, optimBufCTX[p.first]->desc.cudnnDesc,
                         optimBufCTX[p.first]->data->data, &val1))
        checkCUDNN(cudnnScaleTensor(cudnnHdlG, optimBufCTX[p.first]->desc.cudnnDesc,
                         optimBufCTX[p.first]->grad->data, &val2))
                         
        //update
        AdamA(p.second->desc.dType, p.second->data->data, optimBufCTX[p.first]->data->data,
              optimBufCTX[p.first]->grad->data, EPSILON, LEARNING_RATE, L2, p.second->desc.numel);
        
        t++;
        
        //reset backward
        cudaMemset(p.second->grad->data, 0, p.second->data->memSize);
        assertCuda(__FILE__, __LINE__);
    }
}

void dylann::Adam::zeroGrad() {
    for (auto p : optimBufCTX) {
        cudaMemset(p.second->data->data, 0, p.second->data->memSize);
        cudaMemset(p.second->grad->data, 0, p.second->data->memSize);
        assertCuda(__FILE__, __LINE__);
    }

}
