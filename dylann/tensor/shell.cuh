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
    
    template<typename T>
    cuTensor randUniform(cuTensor& A, T min, T max){
        return A.template randUniform(min, max);
    }
    
    template<typename T>
    cuTensor randNormal(cuTensor& A, T mean, T stddev){
        return A.template randNormal(mean, stddev);
    }
}


#endif //DYLANN_SHELL_CUH
