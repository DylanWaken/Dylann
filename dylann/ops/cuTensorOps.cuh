//
// Created by Dylan on 8/5/2022.
//

#ifndef DYLANN_CUTENSOROPS_CUH
#define DYLANN_CUTENSOROPS_CUH

#include "../tensor/cuTensorBase.cuh"
#include "../ops/opRegistry.cuh"
#include <curand.h>
#include <curand_kernel.h>


namespace dylann{
    
    //add operation (for data)
    cuTensorBase* add(cuTensorBase* A, cuTensorBase* B, float alpha, float beta);
    
    //multiply by constant (data)
    cuTensorBase* scale(cuTensorBase* A, float alpha);
    
    cuTensorBase* randUniformOp(cuTensorBase* A, double min, double max);
    
    cuTensorBase* randNormalOp(cuTensorBase* A, double mean, double stddev);
    
    //debug function
    //randomly / constantly fill gradient to test gradient computation
    cuTensorBase* randNormalGradOp(cuTensorBase* A, double mean, double stddev);
}


#endif //DYLANN_CUTENSOROPS_CUH
