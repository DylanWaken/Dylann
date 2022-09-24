//
// Created by Dylan on 9/23/2022.
//

#include "Data.cuh"

HostTensorBase *HostTensorBase::create(dylann::shape4 sizes, cudnnDataType_t dataType) {
    HostTensorBase* tensor;
    cudaMallocHost(&tensor, sizeof(HostTensorBase));
    assertCuda(__FILE__, __LINE__);
    
    tensor->sizes = sizes;
    tensor->dataType = dataType;
    tensor->elemSize = dylann::sizeOfDtype(dataType);
    
    cudaMallocHost(&tensor->data, tensor->sizes.size * tensor->elemSize);
    assertCuda(__FILE__, __LINE__);
    
    return tensor;
}
