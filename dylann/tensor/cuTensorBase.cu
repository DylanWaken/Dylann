//
// Created by Dylan on 8/4/2022.
//

#include "cuTensorBase.cuh"

cudnnHandle_t cudnnHdlG;
cublasHandle_t cublasHdlG;
void* cudnnWorkspaceG;

namespace dylann {
    unsigned int* tensorIDSeqG;
    unsigned int tensorIDGlobal;
    bool onModelRegisterG = false;
    
    //some definitions
    map <TENSOR_PTR, cuTensorBase*>* cuTensorBase::tensorPoolG;
    
    uint64_t sizeOfDtype(cudnnDataType_t dtype){
        switch(dtype){
            case CUDNN_DATA_FLOAT: return sizeof(float);
            case CUDNN_DATA_DOUBLE: return sizeof(double);
            case CUDNN_DATA_HALF: return sizeof(half);
            case CUDNN_DATA_INT8: return sizeof(int8_t);
            default:
                throw std::runtime_error("unsupported dtype");
        }
    }
    
    void assertOnSameDev(std::initializer_list<cuTensorBase*> list){
        assert(list.size() > 0);
        auto dev = (*list.begin())->data->deviceID;
        for(auto it = list.begin() + 1; it != list.end(); ++it){
            if((*it)->data->deviceID != dev){
                logFatal(io::LOG_SEG_COMP, "tensors not on same device");
                assert(false);
            }
        }
    }
    
    void assertAllocated(std::initializer_list<cuTensorBase*> list){
        assert(list.size() > 0);
        for(auto it : list){
            if(!it->desc.isAllocated){
                logFatal(io::LOG_SEG_COMP, "tensor not allocated");
                assert(false);
            }
        }
    }
} // dylann