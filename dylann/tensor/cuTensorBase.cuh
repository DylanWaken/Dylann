//
// Created by Dylan on 8/4/2022.
//

#ifndef DYLANN_CUTENSORBASE_CUH
#define DYLANN_CUTENSORBASE_CUH

//#include <torch/torch.h>
#include <cudnn.h>
#include <string>
#include <cuda_fp16.h>
#include "../../cudautil/assertion.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>


#define checkCUDNN(expression){                              \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      logFatal(io::LOG_SEG_COMP, "Cudnn failed, error : ");  \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      assert(false);                                         \
    }                                                        \
}

#define checkCUBLAS(expression){ \
    cublasStatus_t status = (expression); \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
      logFatal(io::LOG_SEG_COMP, "Cublas failed, error : ");  \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cublasGetStatusString(status) << std::endl; \
      assert(false);                                         \
    }                                                        \
}

#define CUDNN_WORKSPACE_SIZE_G (1024 * 1024 * 256)
typedef unsigned int TENSOR_PTR;

using namespace std;
using namespace io;
extern cudnnHandle_t cudnnHdlG;
extern cublasHandle_t cublasHdlG;
extern void* cudnnWorkspaceG;

namespace dylann {
    extern unsigned int* tensorIDSeqG;
    extern unsigned int tensorIDGlobal;
    extern bool onModelRegisterG;
    
    uint64_t sizeOfDtype(cudnnDataType_t dtype);
    
    /**
     * @Brief index4 is used to navigate within A 4D tensor
     * Tensors are stored in row major order (NCHW)
     *
     * //NHWC is not supported because its A cursed arrangement
     * that should be burned to death in the flame of hell
     *
     * n : the 4th dimension
     * c : channels
     * h : rows (height)
     * w : cols (width)
     */
    struct index4{
        uint64_t n=0, c=0, h=0, w=0;
        __device__ __host__ index4(uint64_t n, uint64_t c, uint64_t h, uint64_t w) : n(n), c(c), h(h), w(w){}
        __device__ __host__ index4(uint64_t c, uint64_t h, uint64_t w) : c(c), h(h), w(w){}
        __device__ __host__ index4(uint64_t h, uint64_t w) : h(h), w(w){}
        __device__ __host__ explicit index4(uint64_t w) : w(w){}
        
        [[nodiscard]] __device__ __host__ uint64_t getOffset() const{
            
        }
        [[nodiscard]] string toString() const{
            return "(" + to_string(n) + "," + to_string(c) +
                   "," + to_string(h) + "," + to_string(w) + ")";
        }
    };
    
    /**
     * shape of tensors
     */
    struct shape4 : public index4{
        uint64_t size;
        __device__ __host__ shape4(uint64_t n, uint64_t c, uint64_t h, uint64_t w)
                : index4(n, c, h, w){ size = n*c*h*w; }
        
        __device__ __host__ shape4(uint64_t c, uint64_t h, uint64_t w)
                : index4(1, c, h, w){ size = c*h*w; }
        
        __device__ __host__ shape4(uint64_t h, uint64_t w)
                : index4(1, 1, h, w){ size = h*w; }
                
        __device__ __host__ explicit shape4(uint64_t w)
                : index4(1, 1, 1, w){ size = w; }
    };
    
    struct TDescriptor{
    public:
        //desc
        cudnnTensorDescriptor_t cudnnDesc{};
        
        cudnnDataType_t dType;
        shape4 sizes;
        uint64_t numel;
        uint64_t elementSize;
        
        uint64_t uuid;
        
        //this is used as a key
        //when the recursive backwardCalc function is called with branching network
        //only backwardRecur() call from the tensor with key would continue the recursion
        //some models like resnet and densenet depends heavily on this feature
        uint64_t gradSrcUuid;
        
        //state
        bool isAllocated = false;
        bool withGrad = false;
        bool isParam = false;
        
        static TDescriptor* create(shape4 dims, cudnnDataType_t dType) {
            //create the global cudnn Handle
            if(cudnnHdlG == nullptr){
                cudnnCreate(&cudnnHdlG);
            }
            
            //create the global cublas Handle
            if(cublasHdlG == nullptr){
                cublasCreate(&cublasHdlG);
            }
            
            //create the global cudnn workspace
            if(cudnnWorkspaceG == nullptr){
                cudaMalloc(&cudnnWorkspaceG, CUDNN_WORKSPACE_SIZE_G);
            }
            
            TDescriptor* desc;
            cudaMallocHost(&desc, sizeof(TDescriptor));
    
            desc->dType = dType;
            
            //we assign a unique id to each tensor in the model
            if(onModelRegisterG) {
                desc->uuid = *tensorIDSeqG;
                (*tensorIDSeqG)++;
            }else{
                desc->uuid = tensorIDGlobal;
                tensorIDGlobal++;
            }
            
            //initialize the cudnn tensor cudnnDesc
            cudnnCreateTensorDescriptor(&desc->cudnnDesc);
            cudnnSetTensor4dDescriptor(desc->cudnnDesc,
                                       CUDNN_TENSOR_NCHW,
                                       dType,
                                       (int) dims.n,
                                       (int) dims.c,
                                       (int) dims.h,
                                       (int) dims.w);
    
            desc->numel = dims.size;
            desc->elementSize = sizeOfDtype(dType);
            desc->sizes = dims;
            
            return desc;
        }
        
        void reshape(shape4 dims){
            assert(dims.size == this->sizes.size);
            cudnnSetTensor4dDescriptor(cudnnDesc,
                                       CUDNN_TENSOR_NCHW,
                                       dType,
                                       (int) dims.n,
                                       (int) dims.c,
                                       (int) dims.h,
                                       (int) dims.w);
            this->sizes = dims;
        }
    };
    
    struct TStorage{
        void* data;
        int deviceID;
        uint64_t memSize;
        
        TStorage() = default;
        
        static TStorage* create(int deviceID, uint64_t memSize){
            
            TStorage* storage;
            cudaMallocHost(&storage, sizeof(TStorage));
            
            storage->deviceID = deviceID;
            storage->memSize = memSize;
            
            cudaSetDevice(deviceID);
            cudaMalloc(&storage->data, memSize);
            cudaMemset(storage->data, 0, memSize);
            assertCuda(__FILE__, __LINE__);
            
            return storage;
        }
    };
    
    struct cuTensorBase {
    public:
        TDescriptor desc;
        TStorage* data{};
        TStorage* grad{};
        static map<TENSOR_PTR ,cuTensorBase*>* tensorPoolG;
        
        static cuTensorBase* create(shape4 dims, cudnnDataType_t dType){
            cuTensorBase* tensor;
            cudaMallocHost(&tensor, sizeof(cuTensorBase));
            
            tensor->desc = *TDescriptor::create(dims, dType);
            if(onModelRegisterG) tensorPoolG->insert({tensor->desc.uuid, tensor});
            
            return tensor;
        }
        
        cuTensorBase* instantiate(int deviceID){
            this->data = TStorage::create(deviceID, this->desc.numel * this->desc.elementSize);
            return this;
        }
    };
    
    
    void assertOnSameDev(std::initializer_list<cuTensorBase*> list);
    void assertAllocated(std::initializer_list<cuTensorBase*> list);
    
} // dylann

#endif //DYLANN_CUTENSORBASE_CUH
