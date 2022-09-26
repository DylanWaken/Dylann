//
// Created by Dylan on 9/24/2022.
//

#ifndef DYLANN_BUILDINREADFUNCS_CUH
#define DYLANN_BUILDINREADFUNCS_CUH

#include <utility>
#include <fstream>
#include <iostream>

#include "../../io/dataset/Dataset.cuh"

namespace io{
    struct CIFAR_10ReadFunc : public ReadFuncCV{
    public:
        string path;
        mutex lock;
        int CPU_WORKER_COUNT;
        unsigned char* fileBuffer = nullptr;
        
        CIFAR_10ReadFunc(string path, int readThreads) : path(std::move(path)), CPU_WORKER_COUNT(readThreads){
            cudaMallocHost(&fileBuffer, 32 * 32 * 3 * CPU_WORKER_COUNT);
            assertCuda(__FILE__, __LINE__);
        }
        
        cv::Mat readNxt(unsigned int sampleID, int tid, int tc, bool istest) override;
        
        void readNxtLabel(unsigned int batchSampleID, unsigned int sampleID, Data& data,
                          int tid, int tc, bool istest) override;
    };
}

#endif //DYLANN_BUILDINREADFUNCS_CUH
