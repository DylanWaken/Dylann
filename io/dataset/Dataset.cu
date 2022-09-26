//
// Created by Dylan on 9/24/2022.
//

#include "Dataset.cuh"

namespace io {
    
    void Dataset::nextMiniBatch(cuTensorBase *X, cuTensorBase *Y) {
        vector<Data>* usingSet = &data[ramLoadID % 2];
        
        //async data preloading
        if (indexes.empty()) {
            //wait for the next batch to be loaded
            while(isDataFetching){}
            
            //launch the next batch//
            thread(&Dataset::launchPipeline, this, ref(*usingSet)).detach();
            
            //change to the preloaded batch
            ramLoadID += 1;
            usingSet = &data[ramLoadID % 2];
            
            for(int i = 0; i < RAM_BATCH_SIZE; i++){
                indexes.push_back(i);
            }
        }
        
        //randomized
        default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
        
        //copy data to GPU
        for (int i = 0; i < MINI_BATCH_SIZE; i++) {
            uniform_int_distribution<int> distribution(0, (int)indexes.size()-1);
            
            size_t xOffset = i * X->desc.elementSize * X->desc.sizes.size / MINI_BATCH_SIZE;
            size_t yOffset = i * Y->desc.elementSize * Y->desc.sizes.size / MINI_BATCH_SIZE;
            int index = distribution(generator);
            int randSample = indexes[index];
            
            void* copyPtrX = (*usingSet)[randSample].X->data;
    
            assertCuda(__FILE__, __LINE__);
            cudaMemcpy((unsigned char *) X->data->data + xOffset,
                       copyPtrX, X->desc.elementSize * X->desc.sizes.size / MINI_BATCH_SIZE,
                       cudaMemcpyHostToDevice);
            assertCuda(__FILE__, __LINE__);
            
            cudaMemcpy((unsigned char *) Y->data->data + yOffset,
                       (*usingSet)[randSample].Y->data, Y->desc.elementSize * Y->desc.sizes.size / MINI_BATCH_SIZE,
                       cudaMemcpyHostToDevice);
            assertCuda(__FILE__, __LINE__);
    
            indexes.erase(indexes.begin() + index);
        }
    }
    
    Dataset::Dataset(unsigned int EPOCH_SIZE, unsigned int RAM_BATCH_SIZE, unsigned int MINI_BATCH_SIZE,
                     unsigned int CPU_WORKERS, unsigned int VAL_BATCH_SIZE, shape4 dataShape, shape4 labelShape, cudnnDataType_t dataType) :
                     EPOCH_SIZE(EPOCH_SIZE), RAM_BATCH_SIZE(RAM_BATCH_SIZE), MINI_BATCH_SIZE(MINI_BATCH_SIZE),
                     CPU_WORKERS(CPU_WORKERS), dataShape(dataShape), labelShape(labelShape), dataType(dataType),
                     VAL_BATCH_SIZE(VAL_BATCH_SIZE) {
        
        logInfo(LOG_SEG_IO, "Registered Dataset: EPOCH " + to_string(EPOCH_SIZE) + " RAM_BATCH " + to_string(RAM_BATCH_SIZE) +
                            " MINI_BATCH " + to_string(MINI_BATCH_SIZE) + " CPU_WORKERS " + to_string(CPU_WORKERS) +
                            " VAL_BATCH " + to_string(VAL_BATCH_SIZE));
    }
    
    void Dataset::construct() {
    
        for (int i = 0; i < RAM_BATCH_SIZE; i++) {
            data[0].push_back(*Data::create(dataShape, labelShape, dataType));
            data[1].push_back(*Data::create(dataShape, labelShape, dataType));
        }
        
        for (int i = 0; i < RAM_BATCH_SIZE; i++) {
            valid.push_back(*Data::create(dataShape, labelShape, dataType));
        }
        
        //generate validation set
        fetchValSet();
        launchPipeline(data[0]);
    
        //launch the next batch//
        thread(&Dataset::launchPipeline, this, ref(data[1])).detach();
        for(int i = 0; i < RAM_BATCH_SIZE; i++){
            indexes.push_back(i);
        }
        ramLoadID = 0;
        logInfo(LOG_SEG_IO, "Dataset constructed");
    }
    
    void DatasetCV::launchPipeline(vector<Data>& data) {
        isDataFetching = true;
        
        _alloc((int)CPU_WORKERS, pipelineCV, readFunc, ref(augCV), ref(augTensor), ref(data), RAM_BATCH_SIZE,
               globalSampleID, EPOCH_SIZE, false);
        globalSampleID += RAM_BATCH_SIZE;
        globalSampleID %= EPOCH_SIZE;
        
        isDataFetching = false;
    }
    
    void DatasetCV::fetchValSet() {
        unsigned int sampleID = 0;
    
        _alloc((int)CPU_WORKERS, pipelineCV, readFunc, ref(augCV), ref(augTensor), ref(valid), VAL_BATCH_SIZE,
                sampleID, EPOCH_SIZE, true);
        logDebug(LOG_SEG_IO,"Validation set fetched");
    }
}