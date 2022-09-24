//
// Created by Dylan on 9/23/2022.
//

#ifndef DYLANN_DATAPIPELINE_CUH
#define DYLANN_DATAPIPELINE_CUH

#include <opencv2/opencv.hpp>
#include <vector>
#include "AugCVInstructions.cuh"
#include "Data.cuh"

#include <thread>
#include <mutex>
#include <condition_variable>


using namespace std;
namespace io {
    
    struct ReadFuncCV{
        virtual cv::Mat readNxt(unsigned int sampleID) = 0;
        virtual void readNxtLabel( unsigned int seq, unsigned int sampleID, Data* data) = 0;
    };
    
    void pipelineCV(int tid, int tc, ReadFuncCV* readFunc, vector<AugmentIns*> augIns, Data* data, int sampleCount, int begin);
    
} // io

#endif //DYLANN_DATAPIPELINE_CUH
