//
// Created by Dylan on 9/24/2022.
//

#ifndef DYLANN_DATASET_CUH
#define DYLANN_DATASET_CUH


#include "Data.cuh"
#include "AugCVInstructions.cuh"
#include "AngTensorInstructions.cuh"
#include "DataPipeline.cuh"
#include "../../cudautil/ThreadController.cuh"
#include <random>

using namespace std;
using namespace dylann;
namespace io {
    class Dataset{
    public:
        //prefetching design
        vector<Data> data[2];
        vector<Data> valid;
        vector<int> indexes;
        
        unsigned int EPOCH_SIZE;
        unsigned int RAM_BATCH_SIZE;
        unsigned int MINI_BATCH_SIZE;
        unsigned const int CPU_WORKERS;
        unsigned int VAL_BATCH_SIZE;
        
        shape4 dataShape;
        shape4 labelShape;
        cudnnDataType_t dataType;
        unsigned int epochID;
        
        //loading data in batches to ram for maximizing CPU utilization
        unsigned int ramLoadID = 0;
        unsigned int globalSampleID = 0;
        unsigned int valSampleID = 0;
        
        bool isDataFetching = false;
        
        Dataset(unsigned int EPOCH_SIZE, unsigned int RAM_BATCH_SIZE, unsigned int MINI_BATCH_SIZE,
                unsigned int CPU_WORKERS, unsigned int VAL_BATCH_SIZE, shape4 dataShape, shape4 labelShape, cudnnDataType_t dataType);
        
        virtual void construct();
        
        //store next mini batch of data in X and Y
        void nextMiniBatch(cuTensorBase* X, cuTensorBase* Y);
        
        void nextValBatch(cuTensorBase* X, cuTensorBase* Y);
        
        virtual void launchPipeline(vector<Data>& data) = 0;
        
        virtual void fetchValSet() = 0;
    };
    
    class DatasetCV : public Dataset{
    public:
        vector<AugmentInsCV*> augCV;
        vector<AugmentInsTensor*> augTensor;
        ReadFuncCV* readFunc{};
    
        DatasetCV(unsigned int EPOCH_SIZE, unsigned int RAM_BATCH_SIZE, unsigned int MINI_BATCH_SIZE,
                unsigned int CPU_WORKERS, unsigned int VAL_BATCH_SIZE,  shape4 dataShape, shape4 labelShape, cudnnDataType_t dataType):
                Dataset(EPOCH_SIZE, RAM_BATCH_SIZE, MINI_BATCH_SIZE, CPU_WORKERS, VAL_BATCH_SIZE, dataShape, labelShape, dataType){
        }
        
        void bindAugCV(initializer_list<AugmentInsCV*> augCVIn){
            this->augCV = augCVIn;
        }
        
        void bindAugTensor(initializer_list<AugmentInsTensor*> augTensorIn){
            this->augTensor = augTensorIn;
        }
        
        void bindReadFunc(ReadFuncCV* readFuncIn){
            this->readFunc = readFuncIn;
        }
        
        void launchPipeline(vector<Data>& data) override;
        
        void fetchValSet() override;
    };
}


#endif //DYLANN_DATASET_CUH
