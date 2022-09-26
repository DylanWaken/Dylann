//
// Created by Dylan on 8/6/2022.
//

#include "cuTensor.cuh"
#include <cstdio>
#include <fstream>
#define h2f(x) ((x&0x8000)<<16) | (((x&0x7c00)+0x1C000)<<13) | ((x&0x03FF)<<13)


namespace dylann{
    vector<Operation*>* cuTensor::instructions;
    
    cuTensor cuTensor::operator+=(cuTensor& A) {
        addOp(this->impl, A.impl, 1, 1);
        
        auto* op = new ADD(this->impl->desc.uuid, A.impl->desc.uuid, 1, 1);
        instructions->push_back(op);
        
        return *this;
    }
    
    cuTensor cuTensor::operator-=(cuTensor &other) {
        addOp(this->impl, other.impl, 1, -1);
    
        auto* op = new ADD(this->impl->desc.uuid, other.impl->desc.uuid, 1, -1);
        instructions->push_back(op);
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
                        + w] << ", ";
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
                        cout << val << ", ";
                    }
                    cout << endl;
                }
                cout << endl;
            }
            cout << endl;
        }
    }
    
    template<typename T>
    inline void toFileLoop(void* data, TDescriptor& desc, ofstream& file){
        T* ptr = (T*)data;
        for(auto index = 0 ; index < desc.numel; index ++){
            file << ptr[index] << ", ";
        }
    }
    
    inline void toFileHalf(void* data, TDescriptor& desc, ofstream& file){
        half* ptr = (half*)data;
        for(auto index = 0 ; index < desc.numel; index ++){
            file << __half2float(ptr[index]) << ", ";
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
            default:
                throw std::runtime_error("unsupported dtype");
        }
    }
    
    inline void toFileElement(void* data, TDescriptor& desc, ofstream& file){
        switch (desc.dType) {
            case CUDNN_DATA_FLOAT:
                toFileLoop<float>(data, desc, file);
                break;
            case CUDNN_DATA_DOUBLE:
                toFileLoop<double>(data, desc, file);
                break;
            case CUDNN_DATA_HALF:
                toFileHalf(data, desc, file);
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
    
    void cuTensor::toFile(const char * basePath) {
        if (!impl->desc.isAllocated) return;
        void* view;
        
        ofstream dataFile = ofstream(basePath + to_string(impl->desc.uuid) + string("data.txt"));
        ofstream gradFile = ofstream(basePath + to_string(impl->desc.uuid) + string("backward.txt"));
        
        cudaMallocHost(&view, impl->data->memSize);
        assertCuda(__FILE__, __LINE__);
        cudaSetDevice(impl->data->deviceID);
    
        cudaMemcpy(view, impl->data->data, impl->data->memSize, cudaMemcpyDeviceToHost);
        assertCuda(__FILE__, __LINE__);
        toFileElement(view, impl->desc, dataFile);
        
        if (impl->desc.withGrad) {
            cudaMemcpy(view, impl->grad->data, impl->grad->memSize, cudaMemcpyDeviceToHost);
            assertCuda(__FILE__, __LINE__);
            toFileElement(view, impl->desc, gradFile);
        }
        
        cudaFreeHost(view);
    }
    
}