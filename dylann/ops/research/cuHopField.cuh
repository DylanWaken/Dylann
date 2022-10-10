//
// Created by Dylan on 10/8/2022.
//

#ifndef DYLANN_CUHOPFIELD_CUH
#define DYLANN_CUHOPFIELD_CUH

#include "../../tensor/cuTensorBase.cuh"

//Note: research kernels are just for experiments, they are not optimized for efficiency
//research kernels should not be used in production code

namespace dylann {
    //update an undirected hopfield network with a batch of inputs
    //W is the adjacency matrix, SRef is the batch of information to store
    //the network is updated through : W[i,j] = W[i,j] + 4 * (SRef[i] - 1/2) * (SRef[j] - 1/2)
    
    //network S range is [0,1] (Binary)
    cuTensorBase* updateHopFieldOp(cuTensorBase* W, cuTensorBase* SRef);
    
    //network S range is [-1,1] (Binary)
    cuTensorBase* updateHopFieldNOp(cuTensorBase* W, cuTensorBase* SRef);
    
    //retrive the stored information from the network
    //this is a single threaded application so we use cpu for it.
    cuTensorBase* retrieveHopFieldOp(cuTensorBase* W, cuTensorBase* SAcc, void* WHostBuf, void* SAccHostBuf);
    
    cuTensorBase* randNoiseOp(cuTensorBase* S, float p);
    
    cuTensorBase* val2binOp(cuTensorBase* S, float bin1, float bin2);
}

#endif //DYLANN_CUHOPFIELD_CUH
