//
// Created by Dylan on 8/6/2022.
//

#include "cuTensor.cuh"
#define h2f(x) ((x&0x8000)<<16) | (((x&0x7c00)+0x1C000)<<13) | ((x&0x03FF)<<13)


namespace dylann{
    cuTensor cuTensor::operator+=(cuTensor& A) {
        addOp(this->impl, A.impl, 1, 1);
        GradTracker* t1 = new GRAD_ADD_A(1);
        this->gradStack.emplace(this,t1);
        
        //Tensor “A” here means the tensor B in the original addOp operation
        GradTracker* t2 = new GRAD_ADD_B(1, A.impl);
        this->gradStack.emplace(&A, t2);
        return *this;
    }
    
    cuTensor cuTensor::operator-=(cuTensor &other) {
        addOp(this->impl, other.impl, 1, -1);
        GradTracker* t1 = new GRAD_ADD_A(1);
        this->gradStack.emplace(this, t1);
        
        //Tensor “A” here means the tensor B in the original addOp operation
        GradTracker* t2 = new GRAD_ADD_B(-1, other.impl);
        this->gradStack.emplace(&other, t2);
        return *this;
    }
    
    template<typename T>
    inline void printLoop(void *data, TDescriptor& desc){
        T* ptr = (T*)data;
        for(auto n = 0 ; n < desc.sizes.n; n ++){
            for(auto c = 0 ; c < desc.sizes.c; c ++){
                for(auto h = 0 ; h < desc.sizes.h; h ++){
                    for(auto w = 0 ; w < desc.sizes.w; w ++){
                        cout << (double) ptr[n*desc.sizes.c*desc.sizes.h*desc.sizes.w
                        + c*desc.sizes.h*desc.sizes.w
                        + h*desc.sizes.w
                        + w] << " ";
                    }
                    cout << endl;
                }
                cout << endl;
            }
            cout << endl;
        }
    }
    
    inline void printHalf(void *data, TDescriptor& desc){
        half* ptr = (half*)data;
        for(auto n = 0 ; n < desc.sizes.n; n ++){
            for(auto c = 0 ; c < desc.sizes.c; c ++){
                for(auto h = 0 ; h < desc.sizes.h; h ++){
                    for(auto w = 0 ; w < desc.sizes.w; w ++){
                        float val = __half2float(ptr[n*desc.sizes.c*desc.sizes.h*desc.sizes.w
                                                     + c*desc.sizes.h*desc.sizes.w
                                                     + h*desc.sizes.w
                                                     + w]);
                        cout << val << " ";
                    }
                    cout << endl;
                }
                cout << endl;
            }
            cout << endl;
        }
    }
    
    inline void printElement(void* data, TDescriptor& desc){
        switch (desc.dType) {
            case CUDNN_DATA_FLOAT:
                printLoop<float>(data, desc);
                break;
            case CUDNN_DATA_DOUBLE:
                printLoop<double>(data, desc);
                break;
            case CUDNN_DATA_HALF:
                printHalf(data, desc);
                break;
            case CUDNN_DATA_INT8:
                printLoop<int8_t>(data, desc);
                break;
            case CUDNN_DATA_INT32:
                printLoop<int32_t>(data, desc);
                break;
            case CUDNN_DATA_INT64:
                printLoop<int64_t>(data, desc);
                break;
            case CUDNN_DATA_UINT8:
                printLoop<uint8_t>(data, desc);
                break;
            default:
                throw std::runtime_error("unsupported dtype");
        }
    }
    
    inline string getDtypeName(cudnnDataType_t dType){
        switch (dType) {
            case CUDNN_DATA_FLOAT:
                return "CUDNN_DATA_FLOAT";
            case CUDNN_DATA_DOUBLE:
                return "CUDNN_DATA_DOUBLE";
            case CUDNN_DATA_HALF:
                return "CUDNN_DATA_HALF";
            case CUDNN_DATA_INT8:
                return "CUDNN_DATA_INT8";
            case CUDNN_DATA_INT32:
                return "CUDNN_DATA_INT32";
            case CUDNN_DATA_INT64:
                return "CUDNN_DATA_INT64";
            default:
                throw std::runtime_error("unsupported dtype");
        }
    }
    
    void cuTensor::print() const {

        if (!impl->desc.isAllocated) return;
        void* view;
        cudaMallocHost(&view, impl->data->memSize);
        assertCuda(__FILE__, __LINE__);
        cudaSetDevice(impl->data->deviceID);
        
        cout << "cuTensor: " << getDtypeName(impl->desc.dType) << endl;
        
        cout<<"------ DATA -------"<<endl;
        cudaMemcpy(view, impl->data->data, impl->data->memSize, cudaMemcpyDeviceToHost);
        assertCuda(__FILE__, __LINE__);
        printElement(view, impl->desc);
    
        if (impl->desc.withGrad) {
            cout << "------ GRAD -------" << endl;
            cudaMemcpy(view, impl->grad->data, impl->grad->memSize, cudaMemcpyDeviceToHost);
            assertCuda(__FILE__, __LINE__);
            printElement(view, impl->desc);
        }
        
        cudaFreeHost(view);
    }
    
    //a recursive funtion that loops through the entire network
    void cuTensor::backwardRecur(uint64_t uuid){
        if (!impl->desc.isAllocated) return;
        if (gradStack.empty()) return;
        
        //check is the key matches
        if (uuid != impl->desc.gradSrcUuid
            && uuid != impl->desc.uuid) return;
    
        cudaSetDevice(impl->data->deviceID);
        while (!gradStack.empty()) {
            auto tracker = gradStack.top();
            gradStack.pop();
            tracker.b->backwardCalc(impl);
            tracker.a->backwardRecur(impl->desc.uuid);
        }
    }
    
    void cuTensor::backward() {
        if (!impl->desc.isAllocated) return;
        if (gradStack.empty()) return;
    
        cudaSetDevice(impl->data->deviceID);
        while (!gradStack.empty()) {
            auto tracker = gradStack.top();
            gradStack.pop();
            tracker.b->backwardCalc(impl);
            tracker.a->backwardRecur(impl->desc.uuid); //call the backwardCalc function recursively
        }
    }
}