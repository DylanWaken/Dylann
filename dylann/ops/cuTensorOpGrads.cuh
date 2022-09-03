//
// Created by Dylan on 8/6/2022.
//

#ifndef DYLANN_CUTENSOROPGRADS_CUH
#define DYLANN_CUTENSOROPGRADS_CUH

#include "../tensor/cuTensorBase.cuh"
#include "cuTensorOps.cuh"

namespace dylann{
    
    cuTensorBase* copy(cuTensorBase* A, cuTensorBase* B);
    
    //when running autograd, each operation will push a grad tracker to the tensor's grad stack
    class GradTracker{
    public:
        cuTensorBase* target{};
        
        //Grad flows from current (the tensor holding this tracker object)
        //to target (or current to current for some special operations)
        virtual void backwardCalc(cuTensorBase* current) = 0;
    };
    
    //for primary tensor in add operation (A in A + B)
    class GRAD_ADD_A : public GradTracker{
    public:
        float alpha;
        explicit GRAD_ADD_A(float alpha) : alpha(alpha){}
        
        //this is a scaling operation, so target is Y
        //y = alpha * A + beta * B
        //∂y/∂A = alpha
        //∂C/∂A = ∂C/∂y * ∂y/∂A = alpha * ∂C/∂y
        void backwardCalc(cuTensorBase* Y) override;
    };
    
    //for tensor B in add operation (B in A + B)
    class GRAD_ADD_B : public GradTracker{
    public:
        float beta;
        GRAD_ADD_B(float beta, cuTensorBase* prev) : beta(beta){
            this->target = prev;
        }
        
        //y = alpha * A + beta * B
        //∂y/∂B = beta
        //∂C/∂B = ∂C/∂y * ∂y/∂B = beta * ∂C/∂y
        void backwardCalc(cuTensorBase* Y) override;
    };
    
    class GRAD_SCALE : public GradTracker{
    public:
        float alpha;
        explicit GRAD_SCALE(float alpha) : alpha(alpha){}
        
        //y = alpha * A
        //∂y/∂A = alpha
        //∂C/∂A = ∂C/∂y * ∂y/∂A = alpha * ∂C/∂y
        void backwardCalc(cuTensorBase* Y) override;
    };
}

#endif //DYLANN_CUTENSOROPGRADS_CUH
