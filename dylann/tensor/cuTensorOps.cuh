//
// Created by Dylan on 8/5/2022.
//

#ifndef DYLANN_CUTENSOROPS_CUH
#define DYLANN_CUTENSOROPS_CUH

#include "cuTensor.cuh"


namespace dylann{
    
    cuTensor copy(const cuTensor &A, const cuTensor &B);
    
    cuTensor add(const cuTensor &A, const cuTensor &B, int alpha, int beta);
    cuTensor add(const cuTensor &A, const cuTensor &B, const cuTensor &C, int alpha, int beta);
}

#endif //DYLANN_CUTENSOROPS_CUH
