//
// Created by Dylan on 8/6/2022.
//

#ifndef DYLANN_CUTENSOR_CUH
#define DYLANN_CUTENSOR_CUH

#include <stack>
#include "cuTensorBase.cuh"
#include "../ops/cuTensorOpGrads.cuh"
#include "../ops/cuTensorOps.cuh"


using namespace std;

namespace dylann{
    
    template<typename A, typename B>
    struct Pair{
    public:
        A a;
        B b;
        
        Pair(A a, B b):a(a), b(b){}
    };
    
    //all functions for tensor is here
    struct cuTensor{
    public:
        cuTensorBase* impl;
        
        //the gradient stack for this tensor (for autograd)
        //<cuTensor src, GradTracker operation_label>
        stack<Pair<cuTensor*, GradTracker*>> gradStack;
        
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
            impl->data = new TStorage(deviceID, impl->desc.numel * impl->desc.elementSize);
            impl->desc.isAllocated = true;
            return *this;
        }
        
        cuTensor instantiateGrad(){
            assert(impl->desc.isAllocated);
            impl->grad = new TStorage(impl->data->deviceID,
                                      impl->desc.numel * impl->desc.elementSize);
            impl->desc.withGrad = true;
            return *this;
        }
        
        template<cudnnDataType_t dtype>
        static cuTensor create(shape4 dims, int deviceID){
            auto out = declare<dtype>(dims);
            return out.instantiate(deviceID);
        }
        
        template<cudnnDataType_t dtype, typename... Args>
        static cuTensor create(int deviceID, Args... args){
            auto out = declare<dtype>(args...);
            return out.instantiate(deviceID);
        }
        
        [[nodiscard]] int getDev() const{ return impl->data->deviceID; }
        
        //initializer
        [[nodiscard]] cuTensor randNormal(double mean, double stddev){
            randNormalOp(this->impl, mean, stddev);
            return *this;
        }
        
        [[nodiscard]] cuTensor randUniform(double min, double max){
            randUniformOp(this->impl, min, max);
            return *this;
        }
       
        //state
        [[nodiscard]] bool isAllocated() const{ return impl->desc.isAllocated; }
        [[nodiscard]] bool withGrad() const{ return impl->desc.withGrad; }
        
        //cudnnDesc
        [[nodiscard]] cudnnTensorDescriptor_t cudnnDesc() const{ return impl->desc.cudnnDesc; }
        [[nodiscard]] TDescriptor desc() const{ return impl->desc; }
        [[nodiscard]] cudnnDataType_t dtype() const{ return impl->desc.dType; }
        [[nodiscard]] shape4 sizes() const{ return impl->desc.sizes; }
        [[nodiscard]] uint64_t memSize() const { return impl->data->memSize; }
        [[nodiscard]] uint64_t numel() const{ return impl->desc.numel; }
        [[nodiscard]] uint64_t elementSize() const{ return impl->desc.elementSize; }
        
        //data storage
        [[nodiscard]] TStorage* data() const{ return impl->data; }
        [[nodiscard]] TStorage* grad() const{ return impl->grad; }
        
        //data access
        [[nodiscard]] void* dataPtr() const{ return impl->data->data; }
        [[nodiscard]] void* gradPtr() const{ return impl->grad->data; }
        
        //debug
        void print() const;
        
        //grad backwardCalc
        void backward();
        void backwardRecur(uint64_t uuid);
        
        //operators
        cuTensor operator+=(cuTensor& other);
        cuTensor operator-=(cuTensor& other);
    };
}


#endif //DYLANN_CUTENSOR_CUH
