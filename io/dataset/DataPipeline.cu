//
// Created by Dylan on 9/23/2022.
//

#include "DataPipeline.cuh"

namespace io {
    
    //NHWC to NCHW
    template<typename T>
    void mat2HostTensor(cv::Mat& mat, HostTensorBase* tensor){
        auto* data = (T*) tensor->data;
        for(int i = 0; i < mat.rows; i++){
            for(int j = 0; j < mat.cols; j++){
                for(int k = 0; k < mat.channels(); k++){
                    data[k * mat.cols * mat.rows + i * mat.cols + j] = mat.at<cv::Vec3b>(i, j)[k];
                }
            }
        }
    }
    
    void pipelineCV(int tid, int tc, ReadFuncCV* readFunc, vector<AugmentInsCV*>& augIns,
                    vector<AugmentInsTensor*>& procIns,vector<Data>& data, int sampleCount, int begin, int EPOCH_SIZE, bool istest){
        
        int sampleBeg = (sampleCount / tc) * tid;
        int sampleEnd = tid == tc - 1 ? sampleCount : (sampleCount / tc) * (tid + 1);
        
        for(int sid = sampleBeg; sid < sampleEnd; sid++){
    
            assert(sid < data.size());
            //read the image as mat
            cv::Mat mat = readFunc->readNxt((sid + begin) % EPOCH_SIZE, tid, tc , istest);
            //read label
            readFunc->readNxtLabel(sid, (sid + begin) % EPOCH_SIZE, data[sid], tid, tc, istest);
    
            if(!istest) {
                //augment the image
                for (auto &augIn: augIns) {
                    mat = augIn->augment(mat);
                }
            }
            
            //convert to tensor
            switch (data[sid].X->dataType) {
                case CUDNN_DATA_FLOAT:
                    mat2HostTensor<float>(mat, data[sid].X);
                    break;
                case CUDNN_DATA_DOUBLE:
                    mat2HostTensor<double>(mat, data[sid].X);
                    break;
                case CUDNN_DATA_HALF:
                    mat2HostTensor<half>(mat, data[sid].X);
                    break;
                default:
                    throw runtime_error("Unsupported data type");
            }
            
            //process the tensor
            for(auto& procIn : procIns){
                procIn->process(data[sid]);
            }
        }
    }
} // io