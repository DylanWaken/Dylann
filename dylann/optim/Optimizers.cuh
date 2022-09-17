//
// Created by Dylan on 9/16/2022.
//

#ifndef DYLANN_OPTIMIZERS_CUH
#define DYLANN_OPTIMIZERS_CUH


#include "../tensor/cuTensorBase.cuh"
#include "../DylannContext.cuh"

namespace dylann {
    
    struct OPTIM_BASE {
    public:
        float LEARNING_RATE{};
        float L2 = 0.0001;
        //grads in optimBufCTX can be used as second order momentum
        map<TENSOR_PTR, cuTensorBase*> optimBufCTX;
        map<TENSOR_PTR, cuTensorBase*>* paramsRes;
        
        OPTIM_BASE() = default;
    
        //apply the gradient to the parameters (weights, biases, etc)
        virtual void bindParams(map<TENSOR_PTR, cuTensorBase*>* params){
            paramsRes = params;
        };
        
        virtual void bindDefaultParams(){
            paramsRes = &optimBufCTX;
        }
        
        virtual void apply() = 0;
    
        virtual void zeroGrad() = 0;
    };
    
    //SGD optimizer : Stochastic Gradient Decent, w = w - η * g
    struct SGD : public OPTIM_BASE {
    public:
        explicit SGD(float LEARNING_RATE) : OPTIM_BASE() {
            this->LEARNING_RATE = LEARNING_RATE;
        }
        
        void apply() override;
        
        void zeroGrad() override;
    };
    
    //Momentum : m[t] = m[t-1] * β + (1 - β) * g[t]
    //           w[t] = w[t-1] - η * m[t]
    struct Momentum : public OPTIM_BASE {
    public:
        float BETA = 0.9;
        
        Momentum(float LEARNING_RATE, float BETA) : OPTIM_BASE() {
            this->LEARNING_RATE = LEARNING_RATE;
            this->BETA = BETA;
        }
        
        explicit Momentum(float LEARNING_RATE) : OPTIM_BASE() {
            this->LEARNING_RATE = LEARNING_RATE;
        }
        
        void bindParams(map<TENSOR_PTR, dylann::cuTensorBase *> *params) override;
        
        void bindDefaultParams() override;
        
        void apply() override;
        
        void zeroGrad() override;
    };


    //RMSProp : V[t] = V[t-1] * β + (1 - β) * g[t]^2
    //          w[t] = w[t-1] - η * g[t] / sqrt(V[t] + ε)
    struct RMSProp : public OPTIM_BASE {
    public:
        float BETA = 0.9;
        float EPSILON = 1e-8;
        
        RMSProp(float LEARNING_RATE, float BETA, float EPSILON) : OPTIM_BASE() {
            this->LEARNING_RATE = LEARNING_RATE;
            this->BETA = BETA;
            this->EPSILON = EPSILON;
        }
        
        explicit RMSProp(float LEARNING_RATE) : OPTIM_BASE() {
            this->LEARNING_RATE = LEARNING_RATE;
        }
        
        void bindParams(map<TENSOR_PTR, dylann::cuTensorBase *> *params) override;
        
        void bindDefaultParams() override;
        
        void apply() override;
        
        void zeroGrad() override;
    };
    
    //Adam : m[t] = m[t-1] * β1 + (1 - β1) * g[t]
    //       V[t] = V[t-1] * β2 + (1 - β2) * g[t]^2
    //       w[t] = w[t-1] - η * m[t] / (sqrt(V[t]) + ε)
    struct Adam : public OPTIM_BASE {
    public:
        float BETA1 = 0.9;
        float BETA2 = 0.999;
        float EPSILON = 1e-8;
        int t = 0;
        
        Adam(float LEARNING_RATE, float BETA1, float BETA2, float EPSILON) : OPTIM_BASE() {
            this->LEARNING_RATE = LEARNING_RATE;
            this->BETA1 = BETA1;
            this->BETA2 = BETA2;
            this->EPSILON = EPSILON;
        }
        
        explicit Adam(float LEARNING_RATE) : OPTIM_BASE() {
            this->LEARNING_RATE = LEARNING_RATE;
        }
        
        void bindParams(map<TENSOR_PTR, dylann::cuTensorBase *> *params) override;
        
        void bindDefaultParams() override;
        
        void apply() override;
        
        void zeroGrad() override;
    };
}


#endif //DYLANN_OPTIMIZERS_CUH
