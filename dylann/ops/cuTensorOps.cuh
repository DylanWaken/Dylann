//
// Created by Dylan on 8/5/2022.
//

#ifndef DYLANN_CUTENSOROPS_CUH
#define DYLANN_CUTENSOROPS_CUH

#include "../tensor/cuTensorBase.cuh"
#include <curand.h>
#include <curand_kernel.h>


namespace dylann{
    
    //addOp operation (for data)
    cuTensorBase* addOp(cuTensorBase* A, cuTensorBase* B, float alpha, float beta);
    cuTensorBase* addOpGrad(cuTensorBase* X, cuTensorBase* Y, float alpha, float beta);
    
    //multiply by constant (data)
    cuTensorBase* scale(cuTensorBase* A, float alpha);
    cuTensorBase* scaleOpGrad(cuTensorBase* A, float alpha);
    
    //flatten
    cuTensorBase* flattenOp(cuTensorBase* X, cuTensorBase* Y);
    cuTensorBase* flattenOpGrad(cuTensorBase* X, cuTensorBase* Y);
    
    cuTensorBase* randUniformOp(cuTensorBase* A, double min, double max);
    
    cuTensorBase* randNormalOp(cuTensorBase* A, double mean, double stddev);
    
    //debug function
    //randomly / constantly fill gradient to test gradient computation
    cuTensorBase* randNormalGradOp(cuTensorBase* A, double mean, double stddev);
}


#endif //DYLANN_CUTENSOROPS_CUH
