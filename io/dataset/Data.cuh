//
// Created by Dylan on 9/23/2022.
//

#ifndef DYLANN_DATA_CUH
#define DYLANN_DATA_CUH

#include "../../dylann/tensor/cuTensorBase.cuh"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace dylann;
namespace io {
    struct HostTensorBase {
         shape4 sizes;
         uint64_t elemSize{};
         cudnnDataType_t dataType{};
         void* data{};
         
         static HostTensorBase* create(shape4 sizes, cudnnDataType_t dataType);
    };
    
    cv::Mat hostTensor2Mat(HostTensorBase* tensor);
    
    void mat2HostTensor(cv::Mat& mat, HostTensorBase* tensor);
    
    struct Data{
        HostTensorBase* X;
        HostTensorBase* Y;
        
        Data(HostTensorBase* X, HostTensorBase* Y){
            this->X = X;
            this->Y = Y;
        }
        
        Data(shape4 dataSizes, shape4 labelSizes, cudnnDataType_t dataType){
            this->X = HostTensorBase::create(dataSizes, dataType);
            this->Y = HostTensorBase::create(labelSizes, dataType);
        }
        
        static Data* create(shape4 dataSizes, shape4 labelSizes, cudnnDataType_t dataType){
            Data* data;
            cudaMallocHost(&data, sizeof(Data));
            assertCuda(__FILE__, __LINE__);
            
            data->X = HostTensorBase::create(dataSizes, dataType);
            data->Y = HostTensorBase::create(labelSizes, dataType);
            
            return data;
        }
        
        void free(){
            cudaFreeHost(X->data);
            assertCuda(__FILE__, __LINE__);
            cudaFreeHost(Y->data);
            assertCuda(__FILE__, __LINE__);
            cudaFreeHost(X);
            assertCuda(__FILE__, __LINE__);
            cudaFreeHost(Y);
            assertCuda(__FILE__, __LINE__);
        }
    };
}


#endif //DYLANN_DATA_CUH
