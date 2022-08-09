//
// Created by Dylan on 8/4/2022.
//

#ifndef DYLANN_CUTENSORBASE_CUH
#define DYLANN_CUTENSORBASE_CUH

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
        cudnnTensorDescriptor_t cudnnDesc{};
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
            
            //initialize the cudnn tensor cudnnDesc
            cudnnCreateTensorDescriptor(&cudnnDesc);
            cudnnSetTensor4dDescriptor(cudnnDesc,
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
    
    
    void assertOnSameDev(std::initializer_list<cuTensorBase*> list);
    void assertAllocated(std::initializer_list<cuTensorBase*> list);
    
} // dylann

#endif //DYLANN_CUTENSORBASE_CUH
