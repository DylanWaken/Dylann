#include <iostream>

#include "presets/Resnet.cuh"
#include "dylann/module/Sequence.cuh"

#include <thread>
#include <mutex>
#include <condition_variable>
#include "io/dataset/Dataset.cuh"
#include "presets/readFuncs/BuildinReadfuncs.cuh"

#include "dylann/ops/research/cuHopField.cuh"
#define MINI_BATCH_SIZE 64

using namespace dylann;
using namespace std;
using namespace io;

int main() {
    
    ofstream file("D:\\Projects\\PyCharm\\graphing\\hopfieldNet\\exp.txt");
    
    initEngineContext();
    auto W0 = cuTensor::create(0, CUDNN_DATA_FLOAT, {100, 100});
    auto LossBuf = cuTensor::create(0, CUDNN_DATA_FLOAT, {1,1});
    
    for(uint64_t n = 1; n < 101; n++){
        auto S0 = cuTensor::create(0, CUDNN_DATA_FLOAT, {n, 1, 1, 100})
                .randNormal(0, 1);
        
        val2binOp(S0.impl, 1, -1);
        updateHopFieldOp(W0.impl, S0.impl);
        
        auto SAcc = cuTensor::create(0, CUDNN_DATA_FLOAT, {n, 1, 1, 100});
        MSE loss = MSE(SAcc.impl);
        void* WHostBuf, *SAccHostBuf;
        
        cudaMallocHost(&WHostBuf, W0.impl->data->memSize);
        assertCuda(__FILE__, __LINE__);
        cudaMallocHost(&SAccHostBuf, SAcc.impl->data->memSize);
        assertCuda(__FILE__, __LINE__);
        
        for(int i = 0; i < 100; i += 1){
            float noise = (float)i/100;
            cudaMemcpy(SAcc.impl->data->data, S0.impl->data->data, S0.impl->data->memSize, cudaMemcpyDeviceToDevice);
            assertCuda(__FILE__, __LINE__);
            randNoiseOp(SAcc.impl, noise);
            
            retrieveHopFieldOp( W0.impl, SAcc.impl, WHostBuf, SAccHostBuf);
            float lossVal = loss.loss(S0.impl);
            
            assertCuda(__FILE__, __LINE__);
            file << lossVal / (n * 200.0f) << ", ";
    
//            if(i == 50 && n == 1){ SAcc.print(); S0.print(); exit(0); }
        }
        
        cudaFreeHost(WHostBuf);
        cudaFreeHost(SAccHostBuf);
    
        cudaFree(S0.impl->data->data);
        cudaFree(SAcc.impl->data->data);
        cudaFreeHost(S0.impl);
        cudaFreeHost(SAcc.impl);
        
        W0.randNormal(0,0);
        
        file<<endl;
        tensorsCTX.clear();
    }
}
