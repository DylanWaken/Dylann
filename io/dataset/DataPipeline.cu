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
                    data[i * mat.cols * mat.channels() + j * mat.channels() + k] = mat.at<cv::Vec3b>(i, j)[k];
                }
            }
        }
    }
    
    void pipelineCV(int tid, int tc, ReadFuncCV* readFunc, vector<AugmentIns*> augIns, Data* data, int sampleCount, int begin){
        
        int sampleBeg = (sampleCount / tc) * tid;
        int sampleEnd = tid == tc - 1 ? sampleCount : (sampleCount / tc) * (tid + 1);
        
        for(int sid = sampleBeg; sid < sampleEnd; sid++){
            //read the image as mat
            cv::Mat mat = readFunc->readNxt(tid + begin);
            //read label
            readFunc->readNxtLabel(tid, tid + begin, data);
    
            //augment the image
            for(auto& augIn : augIns){
                mat = augIn->augment(mat);
            }
    
            //convert to tensor
            switch (data->X->dataType) {
                case CUDNN_DATA_FLOAT:
                    mat2HostTensor<float>(mat, data->X);
                    break;
                case CUDNN_DATA_DOUBLE:
                    mat2HostTensor<double>(mat, data->X);
                    break;
                case CUDNN_DATA_HALF:
                    mat2HostTensor<half>(mat, data->X);
                    break;
                case CUDNN_DATA_INT8:
                    mat2HostTensor<int8_t>(mat, data->X);
                    break;
                case CUDNN_DATA_INT32:
                    mat2HostTensor<int32_t>(mat, data->X);
                    break;
                case CUDNN_DATA_INT64:
                    mat2HostTensor<int64>(mat, data->X);
                    break;
                default:
                    break;
            }
        }
    }
} // io