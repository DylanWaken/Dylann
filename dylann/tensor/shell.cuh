//
// Created by Dylan on 8/8/2022.
//

#ifndef DYLANN_SHELL_CUH
#define DYLANN_SHELL_CUH

#include "cuTensor.cuh"

namespace dylann{
    //these are the "operations" defined for cuTensorBase
    //but implemented with "cuTensor" with gradient tracking functionalities
    
    cuTensor add(cuTensor& A, cuTensor& B, float alpha, float beta);
    
    cuTensor scale(cuTensor& A, float alpha);
    
    cuTensor randUniform(cuTensor& A, double min, double max){
        return A.randUniform(min, max);
    }
    
    cuTensor randNormal(cuTensor& A, double mean, double stddev){
        return A.randNormal(mean, stddev);
    }
}


#endif //DYLANN_SHELL_CUH
