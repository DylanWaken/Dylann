//
// Created by Dylan on 9/24/2022.
//

#include "AngTensorInstructions.cuh"

namespace io {
    
    template<typename T>
    void runUniformNormProc(Data &dataIn, float min, float max) {
        T* data = (T*)dataIn.X->data;
        
        T maxVal = (T)-1e15, minVal = (T)1e15;
        for (int index = 0; index < dataIn.X->sizes.size; index++) {
            T val = data[index];
            
            if (val > maxVal) {
                maxVal = val;
            }
            
            if (val < minVal) {
                minVal = val;
            }
        }
        
        T range = maxVal - minVal;
        
        for (int index = 0; index < dataIn.X->sizes.size; index++) {
            T val = data[index];
            data[index] = ((val - minVal) / range) * (max - min) + min;
        }
    }
    
    void runUniformNormProc(Data &dataIn, float min, float max) {
        half* data = (half*)dataIn.X->data;
        
        half maxVal = (half)-1e15, minVal = (half)1e15;
        for (int index = 0; index < dataIn.X->sizes.size; index++) {
            half val = data[index];
            
            if (__half2float(val) > __half2float(maxVal)) {
                maxVal = val;
            }
            
            if (__half2float(val) < __half2float(minVal)) {
                minVal = val;
            }
        }
        
        float range = __half2float(maxVal) - __half2float(minVal);
        
        for (int index = 0; index < dataIn.X->sizes.size; index++) {
            half val = data[index];
            data[index] = __float2half(((__half2float(val) - __half2float(minVal))
                    / range) * (max - min) + min);
        }
    }
    
    void UniformNorm::process(Data &dataIn) {
        switch (dataIn.X->dataType) {
            case CUDNN_DATA_FLOAT:
                runUniformNormProc<float>(dataIn, min, max);
                break;
            case CUDNN_DATA_DOUBLE:
                runUniformNormProc<double>(dataIn, min, max);
                break;
            case CUDNN_DATA_HALF:
                runUniformNormProc(dataIn, min, max);
                break;
            default:
                throw runtime_error("Unsupported data type");
        }
    }
    
    template<typename T>
    void runStdNormProc(Data &dataIn, float mean, float std) {
        T* data = (T*)dataIn.X->data;
        
        T sum = 0;
        for (int index = 0; index < dataIn.X->sizes.size; index++) {
            T val = data[index];
            sum += val;
        }
        
        T meanVal = sum / dataIn.X->sizes.size;
        
        T sumSq = 0;
        for (int index = 0; index < dataIn.X->sizes.size; index++) {
            T val = data[index];
            sumSq += (val - meanVal) * (val - meanVal);
        }
        
        T stdVal = sqrt(sumSq / dataIn.X->sizes.size);
        
        for (int index = 0; index < dataIn.X->sizes.size; index++) {
            T val = data[index];
            data[index] = (val - meanVal) / stdVal * std + mean;
        }
    }
    
    void runStdNormProc(Data &dataIn, float mean, float std) {
        half* data = (half*)dataIn.X->data;
        
        float sum = 0;
        for (int index = 0; index < dataIn.X->sizes.size; index++) {
            half val = data[index];
            sum += __half2float(val);
        }
        
        float meanVal = sum / (float)dataIn.X->sizes.size;
        
        float sumSq = __float2half(0);
        for (int index = 0; index < dataIn.X->sizes.size; index++) {
            half val = data[index];
            sumSq += (__half2float(val) - meanVal) * (__half2float(val) - meanVal);
        }
        
        float stdVal = sqrt(sumSq / (float)dataIn.X->sizes.size);
        
        for (int index = 0; index < dataIn.X->sizes.size; index++) {
            half val = data[index];
            data[index] = __float2half((__half2float(val) - meanVal) / stdVal * std + mean);
        }
    }
    
    void StdNorm::process(Data &dataIn) {
        switch (dataIn.X->dataType) {
            case CUDNN_DATA_FLOAT:
                runStdNormProc<float>(dataIn, mean, std);
                break;
            case CUDNN_DATA_DOUBLE:
                runStdNormProc<double>(dataIn, mean, std);
                break;
            case CUDNN_DATA_HALF:
                runStdNormProc(dataIn, mean, std);
                break;
            default:
                throw runtime_error("Unsupported data type");
        }
    }
} // io