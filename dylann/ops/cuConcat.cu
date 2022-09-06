//
// Created by Dylan on 9/4/2022.
//

#include "cuConcat.cuh"

namespace dylann {
    
    template<typename T>
    __global__ void concatD(cuTensorBase** Xs, int paramCount, cuTensorBase* Y){
        unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
        if(id >= Y->desc.sizes.size) return;
        unsigned int n = Y->desc.sizes.n;
        unsigned int optSize = Y->desc.sizes.size / n;
        unsigned int cIndex = id % optSize;
        unsigned int nIndex = id / optSize;

        cuTensorBase* src;
        unsigned int pId = 0;
        unsigned int shift = 0;
        #pragma unroll
        while(shift + (Xs[pId]->desc.sizes.size / n) <= cIndex){
            shift += (Xs[pId]->desc.sizes.size / n);
            pId++;
            if(pId >= paramCount) return;
        }

        src = Xs[pId];
        unsigned int srcIndex = nIndex * (src->desc.sizes.size / n) + (cIndex - shift);
    
        ((T*)Y->data->data)[id] = ((T*)src->data->data)[srcIndex];
    }
    
    
    cuTensorBase* concatChannelOp(cuTensorBase** Xs, int inputCount, cuTensorBase* Y){

        cudaSetDevice(Xs[0]->data->deviceID);
        
        //check conditions:
        uint64_t c = 0;
        for(auto id = 0; id < inputCount; id++){
            assert(Xs[id]->desc.sizes.n == Y->desc.sizes.n);
            assert(Xs[id]->desc.sizes.h == Y->desc.sizes.h);
            assert(Xs[id]->desc.sizes.w == Y->desc.sizes.w);
            c+= Xs[id]->desc.sizes.c;
        }
        assert(c == Y->desc.sizes.c);

        unsigned int block = 256;
        unsigned int grid = (Y->desc.sizes.size + block - 1) / block;
        switch (Y->desc.dType) {
            case CUDNN_DATA_FLOAT:
                concatD<float><<<grid, block>>>(Xs, inputCount, Y);
                break;
            case CUDNN_DATA_DOUBLE:
                concatD<double><<<grid, block>>>(Xs, inputCount, Y);
                break;
            case CUDNN_DATA_HALF:
                concatD<half><<<grid, block>>>(Xs, inputCount, Y);
                break;
            case CUDNN_DATA_INT8:
                concatD<char><<<grid, block>>>(Xs, inputCount, Y);
                break;
            case CUDNN_DATA_INT32:
                concatD<int><<<grid, block>>>(Xs, inputCount, Y);
                break;
            case CUDNN_DATA_INT64:
                concatD<long long><<<grid, block>>>(Xs, inputCount, Y);
                break;
            default:
                break;
        }

        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
        return Y;
    }
    
    template<typename T>
    __global__ void concatGradsD(cuTensorBase** Xs, int paramCount, cuTensorBase* Y){
        unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
        if(id >= Y->desc.sizes.size) return;
        unsigned int n = Y->desc.sizes.n;
        unsigned int optSize = Y->desc.sizes.size / n;
        unsigned int cIndex = id % optSize;
        unsigned int nIndex = id / optSize;
    
        cuTensorBase* src;
        unsigned int pId = 0;
    
        unsigned int shift = 0;
        #pragma unroll
        while(shift + (Xs[pId]->desc.sizes.size / n) <= cIndex){
            shift += (Xs[pId]->desc.sizes.size / n);
            pId++;
            if(pId >= paramCount) return;
        }
    
        src = Xs[pId];
        unsigned int srcIndex = nIndex * (src->desc.sizes.size / n) + (cIndex - shift);
    
        ((T*)src->grad->data)[srcIndex] += ((T*)Y->grad->data)[id];
    }
    
    void concatChannelOpGrads(cuTensorBase* Y, cuTensorBase** Xs, int inputCount){
        cudaSetDevice(Xs[0]->data->deviceID);
    
        unsigned int block = 512;
        unsigned int grid = (Y->desc.sizes.size + block - 1) / block;
        switch (Y->desc.dType) {
            case CUDNN_DATA_FLOAT:
                concatGradsD<float><<<grid, block>>>(Xs, inputCount, Y);
                break;
            case CUDNN_DATA_DOUBLE:
                concatGradsD<double><<<grid, block>>>(Xs, inputCount, Y);
                break;
            case CUDNN_DATA_HALF:
                concatGradsD<half><<<grid, block>>>(Xs, inputCount, Y);
                break;
            case CUDNN_DATA_INT8:
                concatGradsD<char><<<grid, block>>>(Xs, inputCount, Y);
                break;
            case CUDNN_DATA_INT32:
                concatGradsD<int><<<grid, block>>>(Xs, inputCount, Y);
                break;
            case CUDNN_DATA_INT64:
                concatGradsD<long long><<<grid, block>>>(Xs, inputCount, Y);
                break;
            default:
                break;
        }
    
        cudaDeviceSynchronize();
        assertCuda(__FILE__, __LINE__);
    }
    
    void GRAD_CONCAT_CHANNEL::backwardCalc(cuTensorBase *current) {
        concatChannelOpGrads(current, Xs, inputCount);
    }
} // dylann