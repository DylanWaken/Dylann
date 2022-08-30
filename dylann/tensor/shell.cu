//
// Created by Dylan on 8/8/2022.
//

#include "shell.cuh"

namespace dylann{
    cuTensor add(cuTensor& A, cuTensor& B, float alpha, float beta){
        add(A.impl, B.impl, alpha, beta);
        
        GradTracker* t1 = new GRAD_ADD_A(alpha);
        A.gradStack.emplace(&A, t1);
        
        GradTracker* t2 = new GRAD_ADD_B(beta, B.impl);
        A.gradStack.emplace(&B, t2);
        
        //grad buf is needed in branching structure
        if(!B.withGradBuf()){
            B.instantiateGradBuf();
        }
        
        return A;
    }
    
    cuTensor scale(cuTensor& A, float alpha){
        scale(A.impl, alpha);
        
        GradTracker* t = new GRAD_SCALE(alpha);
        A.gradStack.emplace(&A, t);
    
        return A;
    }
    
    cuTensor linear(cuTensor& W, cuTensor& B, cuTensor& X, cuTensor& Y){
        linearOp(W.impl, B.impl, X.impl, Y.impl);
        
        GradTracker* t1 = new GRAD_LINEAR(W.impl, B.impl, X.impl);
        Y.gradStack.emplace(&X,t1);
        
        return Y;
    }
    
    cuTensor randUniform(cuTensor& A, double min, double max){
        return A.randUniform(min, max);
    }
    
    cuTensor randNormal(cuTensor& A, double mean, double stddev){
        return A.randNormal(mean, stddev);
    }
}
