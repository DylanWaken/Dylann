//
// Created by Dylan on 9/14/2022.
//

#ifndef DYLANN_AUTOGRAD_CUH
#define DYLANN_AUTOGRAD_CUH

#include "GradInstructions.cuh"
/**
 * The core of the engine system
 * automatic gradient formation from the forward instructions
 */
namespace dylann{
    void generateGrads(vector<Operation*> &forwardOps, vector<Operation*> &backwardOps);
}


#endif //DYLANN_AUTOGRAD_CUH
