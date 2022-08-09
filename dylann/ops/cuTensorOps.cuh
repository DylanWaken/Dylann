//
// Created by Dylan on 8/5/2022.
//

#ifndef DYLANN_CUTENSOROPS_CUH
#define DYLANN_CUTENSOROPS_CUH

#include "../tensor/cuTensorBase.cuh"
#include "../ops/opRegistry.cuh"


namespace dylann{
    
    //add operation (for data)
    cuTensorBase* add(cuTensorBase* A, cuTensorBase* B, float alpha, float beta);
    
    //multiply by constant (data)
    cuTensorBase* scale(cuTensorBase* A, float alpha);
}

#endif //DYLANN_CUTENSOROPS_CUH
