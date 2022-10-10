//
// Created by Dylan on 10/8/2022.
//

#include "cuHopField.cuh"
#include <vector>
#include <random>
#include <chrono>
#include <curand_kernel.h>

namespace dylann{

    //NOTE: UNOPTIMIZED KERNELS
    template<typename T>
    __global__ void updateHopFieldD(cuTensorBase* W, cuTensorBase* SRef, T* WData, T* SRefData){
        unsigned int ridX = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ridY = blockIdx.y * blockDim.y + threadIdx.y;
        if (ridX >=  W->desc.sizes.w || ridY >= W->desc.sizes.h) return;
        
        T wVal = WData[ridY * W->desc.sizes.w + ridX];
    
        if (ridX == ridY) {
            WData[ridY * W->desc.sizes.w + ridX] = 0;
            return;
        }
        
        #pragma unroll
        for (int n = 0; n < SRef->desc.sizes.n; ++n) {
            T sVal1 = SRefData[n * SRef->desc.sizes.w + ridX];
            T sVal2 = SRefData[n * SRef->desc.sizes.w + ridY];
            wVal += sVal1 * sVal2;
        }
        
        WData[ridY * W->desc.sizes.w + ridX] = wVal;
    }
    
    template<typename T>
    __global__ void updateHopFieldND(cuTensorBase* W, cuTensorBase* SRef, T* WData, T* SRefData){
        unsigned int ridX = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ridY = blockIdx.y * blockDim.y + threadIdx.y;
        if (ridX >=  W->desc.sizes.w || ridY >= W->desc.sizes.h) return;
        
        T wVal = WData[ridY * W->desc.sizes.w + ridX];
        
        if (ridX == ridY) {
            WData[ridY * W->desc.sizes.w + ridX] = 0;
            return;
        }
        
        #pragma unroll
        for (int n = 0; n < SRef->desc.sizes.n; ++n) {
            T sVal1 = SRefData[n * SRef->desc.sizes.w + ridX];
            T sVal2 = SRefData[n * SRef->desc.sizes.w + ridY];
            wVal += (T)4 * (sVal1 - (T)0.5) * (sVal2 - (T)0.5);
        }
        
        WData[ridY * W->desc.sizes.w + ridX] = wVal;
    }
    
    template<typename T>
    void retriveHopFieldD(cuTensorBase* W, cuTensorBase* SRef, T* WData, T* SAccData){
        
        for (int64_t n = 0; n < SRef->desc.sizes.n; n++) {
            int baseOffset = n * SRef->desc.sizes.w;
            
            vector<int> idSeq;
            for (int i = 0; i < W->desc.sizes.w; ++i) idSeq.push_back(i);
            auto rng = std::default_random_engine(chrono::system_clock::now().time_since_epoch().count());
            shuffle(idSeq.begin(), idSeq.end(), rng);
    
            for (int i = 0; i < W->desc.sizes.w; ++i) {
                int id = idSeq[i];
                T sVal = 0;
        
                //get all connected nodes
                for (int j = 0; j < W->desc.sizes.w; ++j) {
                    sVal += WData[id * W->desc.sizes.w + j] * SAccData[baseOffset + j];
                }
        
                //update the node using binary threshold
                SAccData[baseOffset + id] = sVal > 0 ? 1 : -1;
            }
        }
    }
    
    template <typename T>
    __global__ void randNoiseD(cuTensorBase* S, T* SData, float p, long long seed){
        unsigned int ridX = blockIdx.x * blockDim.x + threadIdx.x;
        curandState_t state;
        curand_init(seed * ridX, ridX, 0, &state);
        if (ridX >= S->desc.sizes.size) return;
        
        float flipFlag = curand_uniform(&state);
        SData[ridX] = curand_uniform(&state) > p ? SData[ridX] : -SData[ridX];
    }
    
    template <typename T>
    __global__ void val2binOpD(cuTensorBase* S, T* SData, float bin1, float bin2){
        unsigned int ridX = blockIdx.x * blockDim.x + threadIdx.x;
        if (ridX >= S->desc.sizes.size) return;
        
        SData[ridX] = SData[ridX] > (T)0 ? bin1 : bin2;
    }
    
    cuTensorBase* updateHopFieldOp(cuTensorBase* W, cuTensorBase* SRef){
        
        dim3 block = dim3(16,16);
        dim3 grid((W->desc.sizes.w + block.x - 1) / block.x, (W->desc.sizes.w + block.y - 1) / block.y);
    
        switch (SRef->desc.dType) {
            case CUDNN_DATA_FLOAT:
                updateHopFieldD<float><<<grid, block>>>(W, SRef, (float*)W->data->data, (float*)SRef->data->data);
                break;
            case CUDNN_DATA_DOUBLE:
                updateHopFieldD<double><<<grid, block>>>(W, SRef, (double*)W->data->data, (double*)SRef->data->data);
                break;
            case CUDNN_DATA_HALF:
                updateHopFieldD<half><<<grid, block>>>(W, SRef, (half*)W->data->data, (half*)SRef->data->data);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return W;
    }
    
    cuTensorBase* updateHopFieldNOp(cuTensorBase* W, cuTensorBase* SRef){
        
        dim3 block = dim3(16,16);
        dim3 grid((W->desc.sizes.w + block.x - 1) / block.x, (W->desc.sizes.w + block.y - 1) / block.y);
        
        switch (SRef->desc.dType) {
            case CUDNN_DATA_FLOAT:
                updateHopFieldND<float><<<grid, block>>>(W, SRef, (float*)W->data->data, (float*)SRef->data->data);
                break;
            case CUDNN_DATA_DOUBLE:
                updateHopFieldND<double><<<grid, block>>>(W, SRef, (double*)W->data->data, (double*)SRef->data->data);
                break;
            case CUDNN_DATA_HALF:
                updateHopFieldND<half><<<grid, block>>>(W, SRef, (half*)W->data->data, (half*)SRef->data->data);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return W;
    }
    
    cuTensorBase* retrieveHopFieldOp(cuTensorBase* W, cuTensorBase* SAcc, void* WHostBuf, void* SAccHostBuf){
        cudaMemcpy(WHostBuf, W->data->data, W->data->memSize, cudaMemcpyDeviceToHost);
        assertCuda(__FILE__, __LINE__);
        cudaMemcpy(SAccHostBuf, SAcc->data->data, SAcc->data->memSize, cudaMemcpyDeviceToHost);
        assertCuda(__FILE__, __LINE__);
        
        switch (W->desc.dType) {
            case CUDNN_DATA_FLOAT:
                retriveHopFieldD<float>(W, SAcc, (float*)WHostBuf, (float*)SAccHostBuf);
                break;
            case CUDNN_DATA_DOUBLE:
                retriveHopFieldD<double>(W, SAcc, (double*)WHostBuf, (double*)SAccHostBuf);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        
        cudaMemcpy(SAcc->data->data, SAccHostBuf, SAcc->data->memSize, cudaMemcpyHostToDevice);
        assertCuda(__FILE__, __LINE__);
        return SAcc;
    }
    
    cuTensorBase* randNoiseOp(cuTensorBase* S, float p){
        unsigned int block = 256;
        dim3 grid((S->desc.sizes.size + block - 1) / block);
        
        switch (S->desc.dType) {
            case CUDNN_DATA_FLOAT:
                randNoiseD<float><<<grid, block>>>(S, (float*)S->data->data, p, chrono::system_clock::now().time_since_epoch().count());
                break;
            case CUDNN_DATA_DOUBLE:
                randNoiseD<double><<<grid, block>>>(S, (double*)S->data->data, p, chrono::system_clock::now().time_since_epoch().count());
                break;
            case CUDNN_DATA_HALF:
                randNoiseD<half><<<grid, block>>>(S, (half*)S->data->data, p, chrono::system_clock::now().time_since_epoch().count());
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return S;
    }
    
    cuTensorBase* val2binOp(cuTensorBase* S, float bin1, float bin2){
        unsigned int block = 256;
        dim3 grid((S->desc.sizes.size + block - 1) / block);
        
        switch (S->desc.dType) {
            case CUDNN_DATA_FLOAT:
                val2binOpD<float><<<grid, block>>>(S, (float*)S->data->data, bin1, bin2);
                break;
            case CUDNN_DATA_DOUBLE:
                val2binOpD<double><<<grid, block>>>(S, (double*)S->data->data, bin1, bin2);
                break;
            case CUDNN_DATA_HALF:
                val2binOpD<half><<<grid, block>>>(S, (half*)S->data->data, bin1, bin2);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return S;
    }
}