//
// Created by Dylan on 8/4/2022.
//

#ifndef DYLANN_CUTENSOR_CUH
#define DYLANN_CUTENSOR_CUH

#include <torch/torch.h>
#include <cudnn.h>
#include <string>
#include "../../cudautil/assertion.cuh"


#define checkCUDNN(expression){                              \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      logFatal(io::LOG_SEG_COMP, "Cudnn failed, error : ");  \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      assert(false);                                         \
    }                                                        \
  }

using namespace std;
using namespace io;
extern cudnnHandle_t cudnnHdlG;
namespace dylann {
    
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
        cudnnTensorDescriptor_t descriptor{};
        cudnnDataType_t dType;
        shape4 sizes;
        uint64_t numel;
        uint64_t elementSize;
        
        //state
        bool isAllocated = false;
        bool withGradBuf = false;
        bool withGrad = false;
        
        TDescriptor(shape4 dims, cudnnDataType_t dType) : sizes(dims) {
            //create the global cudnn Handle
            if(cudnnHdlG == nullptr){
                cudnnCreate(&cudnnHdlG);
            }
            
            this->dType = dType;
            
            //initialize the cudnn tensor descriptor
            cudnnCreateTensorDescriptor(&descriptor);
            cudnnSetTensor4dDescriptor(descriptor,
                                       CUDNN_TENSOR_NCHW,
                                       dType,
                                       (int) dims.n,
                                       (int) dims.c,
                                       (int) dims.h,
                                       (int) dims.w);
            
            this->numel = dims.size;
            this->elementSize = sizeOfDtype(dType);
        }
        
        void reshape(shape4 dims){
            assert(dims.size == this->sizes.size);
            cudnnSetTensor4dDescriptor(descriptor,
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
        
        TStorage(int deviceID, uint64_t memSize) : deviceID(deviceID), memSize(memSize){
            cudaSetDevice(deviceID);
            cudaMalloc(&this->data, memSize);
            assertCuda(__FILE__, __LINE__);
        }
    };
    
    struct cuTensorBase {
    public:
        TDescriptor desc;
        TStorage* data{};
        TStorage* grad{};
        TStorage* gradBuf{};
        
        cuTensorBase(shape4 dims, cudnnDataType_t dType) : desc(dims, dType){}
        
        explicit cuTensorBase(cuTensorBase* other) : desc(other->desc){}
    };
    
    //all functions for tensor is here
    struct cuTensor{
    public:
        cuTensorBase* impl;
        
        template<cudnnDataType_t dtype>
        static cuTensor declare(shape4 dims){
            auto* data = new cuTensorBase(dims, dtype);
            return cuTensor{data};
        }
        
        template<cudnnDataType_t dtype, typename... Args>
        static cuTensor declare(Args... args){
            auto* data = new cuTensorBase(shape4(args...), dtype);
            return cuTensor{data};
        }
        
        cuTensor instantiate(int deviceID){
            impl->data = new TStorage(deviceID, impl->desc.sizes.size*impl->desc.elementSize);
            impl->grad = new TStorage(deviceID, impl->desc.sizes.size*impl->desc.elementSize);
            impl->desc.isAllocated = true;
            impl->desc.withGrad = true;
            return *this;
        }
        
        cuTensor instantiateData(int deviceID){
            impl->data = new TStorage(deviceID, impl->desc.sizes.size*impl->desc.elementSize);
            impl->desc.isAllocated = true;
            return *this;
        }
        
        cuTensor instantiateGrad(){
            assert(impl->desc.isAllocated);
            impl->grad = new TStorage(impl->data->deviceID,
                                      impl->desc.sizes.size*impl->desc.elementSize);
            impl->desc.withGrad = true;
            return *this;
        }
    
        cuTensor instantiateGradBuf(){
            assert(impl->desc.isAllocated);
            assert(impl->desc.withGrad);
            impl->gradBuf = new TStorage(impl->data->deviceID,
                                         impl->desc.sizes.size*impl->desc.elementSize);
            impl->desc.withGradBuf = true;
            return *this;
        }
    
        template<cudnnDataType_t dtype>
        static cuTensor create(shape4 dims, int deviceID){
            auto out = declare<dtype>(dims);
            return out.instantiate(deviceID);
        }
        
        template<cudnnDataType_t dtype, typename... Args>
        static cuTensor create(Args... args, int deviceID){
            auto out = declare<dtype>(args...);
            return out.instantiate(deviceID);
        }
        
        [[nodiscard]] int getDev() const{ return impl->data->deviceID; }
        
        //state
        [[nodiscard]] bool isAllocated() const{ return impl->desc.isAllocated; }
        [[nodiscard]] bool withGrad() const{ return impl->desc.withGrad; }
        [[nodiscard]] bool withGradBuf() const{ return impl->desc.withGradBuf; }
        
        //descriptor
        [[nodiscard]] cudnnTensorDescriptor_t cudnnDesc() const{ return impl->desc.descriptor; }
        [[nodiscard]] TDescriptor desc() const{ return impl->desc; }
        [[nodiscard]] cudnnDataType_t dtype() const{ return impl->desc.dType; }
        [[nodiscard]] shape4 sizes() const{ return impl->desc.sizes; }
        [[nodiscard]] uint64_t memSize() const { return impl->data->memSize; }
        [[nodiscard]] uint64_t numel() const{ return impl->desc.numel; }
        [[nodiscard]] uint64_t elementSize() const{ return impl->desc.elementSize; }
    
        //data storage
        [[nodiscard]] TStorage* data() const{ return impl->data; }
        [[nodiscard]] TStorage* grad() const{ return impl->grad; }
        [[nodiscard]] TStorage* gradBuf() const{ return impl->gradBuf; }
        
        //data access
        [[nodiscard]] void* dataPtr() const{ return impl->data->data; }
        [[nodiscard]] void* gradPtr() const{ return impl->grad->data; }
        [[nodiscard]] void* gradBufPtr() const{ return impl->gradBuf->data; }
        
        //operators
        cuTensor operator+=(cuTensor other);
    };
    
    void assertOnSameDev(std::initializer_list<cuTensor> list);
    void assertAllocated(std::initializer_list<cuTensor> list);
    
} // dylann

#endif //DYLANN_CUTENSOR_CUH
