//
// Created by Dylan on 9/9/2022.
//

#ifndef DYLANN_MODULE_CUH
#define DYLANN_MODULE_CUH

#include "../serial/AutoGrad.cuh"
#include "../tensor/cuTensor.cuh"
#include "../tensor/shell.cuh"

namespace dylann {
    
    /**
     * Module is the base of human interface in programming this thing
     * the defined computations in modules would be translated into instructions
     */
    struct Module {
    public:
        virtual cuTensor forward(cuTensor& X) = 0;
    };
    
} // dylann

#endif //DYLANN_MODULE_CUH
