//
// Created by Dylan on 9/2/2022.
//

#include "cuReduce.cuh"

namespace dylann{
    
    void* oneVec10KF;
    
    template<typename T>
    __global__ void fillVec(T* vec, int size){
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < size){
            vec[idx] = 1;
        }
    }
    
    void initPermanetVecs(){
        if (oneVec10KF == nullptr) {
            cudaMalloc(&oneVec10KF, ONE_VEC_BUF_SIZE);
        }
    }
    
    cuTensorBase* reduceOp(cuTensorBase* X, cuTensorBase* Y, int step){
        initPermanetVecs();
        assert(X->desc.sizes.size % step == 0);
    
        cudaSetDevice(X->data->deviceID);
        
        shape4 sizeSrc = X->desc.sizes;
        void* oneVec = nullptr;
        if(step * X->desc.elementSize <= ONE_VEC_BUF_SIZE){
            oneVec = oneVec10KF;
        }else{
            cudaMalloc(&oneVec, step * X->desc.elementSize);
        }
        
        unsigned int blockSize = 1024;
        unsigned int gridSize = (step + blockSize - 1) / blockSize;
        float af = 1.0f, bf = 0.0f;
        double ad = 1.0f, bd = 0.0f;
        half ah = 1.0f, bh = 0.0f;
        
        switch (X->desc.dType) {
            case CUDNN_DATA_FLOAT:
        
                fillVec<float><<<gridSize, blockSize>>>((float*)oneVec, step);
                cudaDeviceSynchronize();
                assertCuda(__FILE__, __LINE__);
                
                checkCUBLAS(cublasSgemv_v2(
                        cublasHdlG,
                        CUBLAS_OP_T,
                        step,
                        (int)sizeSrc.size / step,
                        &af,
                        (float*)X->data->data,
                        step,
                        (float*)oneVec10KF,
                        1,
                        &bf,
                        (float*)Y->data->data,
                        1
                ))
                break;
            case CUDNN_DATA_DOUBLE:
        
                fillVec<double><<<gridSize, blockSize>>>((double*)oneVec, step);
                cudaDeviceSynchronize();
                assertCuda(__FILE__, __LINE__);
                
                checkCUBLAS(cublasDgemv_v2(
                        cublasHdlG,
                        CUBLAS_OP_T,
                        step,
                        (int)sizeSrc.size / step,
                        &ad,
                        (double*)X->data->data,
                        step,
                        (double*)oneVec10KF,
                        1,
                        &bd,
                        (double*)Y->data->data,
                        1
                ))
                break;
            case CUDNN_DATA_HALF:
                
                fillVec<half><<<gridSize, blockSize>>>((half*)oneVec, step);
                cudaDeviceSynchronize();
                assertCuda(__FILE__, __LINE__);
                
                checkCUBLAS(cublasHgemm(
                        cublasHdlG,
                        CUBLAS_OP_T,
                        CUBLAS_OP_T,
                        (int)sizeSrc.size / step,
                        1,
                        step,
                        &ah,
                        (half*)X->data->data,
                        step,
                        (half*)oneVec10KF,
                        1,
                        &bh,
                        (half*)Y->data->data,
                        step
                ))
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        
        if(oneVec != oneVec10KF){
            cudaFree(oneVec);
        }
    
        return Y;
    }
    
    cuTensorBase* softmaxOp(cuTensorBase* X, cuTensorBase* Y, int step){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
        
        float af = 1.0f, bf = 0.0f;
        shape4 sizeSrc = X->desc.sizes;
    
        assert(X->desc.sizes.size % step == 0);
    
        X->desc.reshape({sizeSrc.size / step, 1,1, static_cast<uint64_t>(step)});
        Y->desc.reshape({sizeSrc.size / step, 1,1, static_cast<uint64_t>(step)});
    
        checkCUDNN(cudnnSoftmaxForward(
                cudnnHdlG,
                CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &af,
                X->desc.cudnnDesc,
                X->data->data,
                &bf,
                Y->desc.cudnnDesc,
                Y->data->data
        ))
        
        X->desc.reshape(sizeSrc);
        Y->desc.reshape(sizeSrc);
        return Y;
    }
    
    cuTensorBase* softmaxOpGrads(cuTensorBase* X, cuTensorBase* Y, int step){
        
        float af = 1.0f, bf = 0.0f;
        shape4 sizeSrc = X->desc.sizes;
        
        assert(X->desc.sizes.size % step == 0);
        
        X->desc.reshape({sizeSrc.size / step, 1,1, static_cast<uint64_t>(step)});
        Y->desc.reshape({sizeSrc.size / step, 1,1, static_cast<uint64_t>(step)});
        
        checkCUDNN(cudnnSoftmaxBackward(
                cudnnHdlG,
                CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &af,
                Y->desc.cudnnDesc,
                Y->data->data,
                Y->desc.cudnnDesc,
                Y->grad->data,
                &bf,
                X->desc.cudnnDesc,
                X->grad->data
        ))
        
        X->desc.reshape(sizeSrc);
        Y->desc.reshape(sizeSrc);
        return X;
    }
    
    
    cuTensorBase* softmaxLogOp(cuTensorBase* X, cuTensorBase* Y, int step){
        assertAllocated({X, Y});
        assertOnSameDev({X, Y});
        
        float af = 1.0f, bf = 0.0f;
        shape4 sizeSrc = X->desc.sizes;
    
        assert(X->desc.sizes.size % step == 0);
    
        X->desc.reshape({sizeSrc.size / step, 1,1, static_cast<uint64_t>(step)});
        Y->desc.reshape({sizeSrc.size / step, 1,1, static_cast<uint64_t>(step)});
    
        checkCUDNN(cudnnSoftmaxForward(
                cudnnHdlG,
                CUDNN_SOFTMAX_LOG,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &af,
                X->desc.cudnnDesc,
                X->data->data,
                &bf,
                Y->desc.cudnnDesc,
                Y->data->data
        ))
        
        X->desc.reshape(sizeSrc);
        Y->desc.reshape(sizeSrc);
        return Y;
    }
    
    cuTensorBase* softmaxLogOpGrads(cuTensorBase* X, cuTensorBase* Y, int step){
        
        float af = 1.0f, bf = 0.0f;
        shape4 sizeSrc = X->desc.sizes;
        
        assert(X->desc.sizes.size % step == 0);
        
        X->desc.reshape({sizeSrc.size / step, 1,1, static_cast<uint64_t>(step)});
        Y->desc.reshape({sizeSrc.size / step, 1,1, static_cast<uint64_t>(step)});
        
        checkCUDNN(cudnnSoftmaxBackward(
                cudnnHdlG,
                CUDNN_SOFTMAX_LOG,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &af,
                Y->desc.cudnnDesc,
                Y->data->data,
                Y->desc.cudnnDesc,
                Y->grad->data,
                &bf,
                X->desc.cudnnDesc,
                X->grad->data
        ))
        
        X->desc.reshape(sizeSrc);
        Y->desc.reshape(sizeSrc);
        return X;
    }
    
    cuTensorBase* softmaxCEOp(cuTensorBase* X, cuTensorBase* Y, int step){
        return softmaxOp(X, Y, step);
    }
    
    cuTensorBase* softmaxCEOpGrads(cuTensorBase* X, cuTensorBase* Y, int step){
        float af = 1.0f, bf = -1.0f;
    
        checkCUDNN(cudnnAddTensor(
                cudnnHdlG,
                &bf,
                Y->desc.cudnnDesc,
                Y->grad->data,
                &af,
                X->desc.cudnnDesc,
                X->grad->data
                ))
                
        return X;
    }
    
    void GRAD_SOFTMAX::backwardCalc(cuTensorBase *Y) {
        softmaxOpGrads(X, Y, step);
    }
    
    void GRAD_SOFTMAX_LOG::backwardCalc(cuTensorBase *Y) {
        softmaxLogOpGrads(X, Y, step);
    }
    
    void GRAD_SOFTMAX_CE::backwardCalc(cuTensorBase *Y) {
        softmaxCEOpGrads(X, Y, step);
    }
}