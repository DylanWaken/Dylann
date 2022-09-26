//
// Created by Dylan on 9/23/2022.
//

#ifndef DYLANN_DATAPIPELINE_CUH
#define DYLANN_DATAPIPELINE_CUH

#include <opencv2/opencv.hpp>
#include <vector>
#include "AugCVInstructions.cuh"
#include "AngTensorInstructions.cuh"
#include "Data.cuh"

#include <thread>
#include <mutex>
#include <condition_variable>


using namespace std;
namespace io {
    
    struct ReadFuncCV{
        virtual cv::Mat readNxt(unsigned int sampleID, int tid, int tc, bool istest) = 0;
        virtual void readNxtLabel(unsigned int batchSampleID, unsigned int sampleID, Data& data,
                                  int tid, int tc, bool istest) = 0;
    };
    
    void pipelineCV(int tid, int tc, ReadFuncCV* readFunc, vector<AugmentInsCV*>& augIns,
                    vector<AugmentInsTensor*>& procIns, vector<Data>& data, int EPOCH_SIZE, int sampleCount,
                    int begin, bool istest);
    
} // io

#endif //DYLANN_DATAPIPELINE_CUH
