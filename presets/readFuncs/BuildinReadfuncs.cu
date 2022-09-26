//
// Created by Dylan on 9/24/2022.
//

#include "BuildinReadfuncs.cuh"

//NCHW to NHWC
void buf2Mat(cv::Mat& mat, const unsigned char* tensor){
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            for(int k = 0; k < mat.channels(); k++){
                 mat.at<cv::Vec3b>(i, j)[k] = tensor[k * mat.rows * mat.cols + i * mat.cols + j];
            }
        }
    }
}

cv::Mat CIFAR_10ReadFunc::readNxt(unsigned int sampleID, int tid, int tc, bool istest) {
    
    int fileID = (int)sampleID / 10000 + 1;
    
    string filePath = this->path + (istest ? "\\test_batch.bin" : "\\data_batch_" + to_string(fileID) + ".bin");
    {
        lock_guard<mutex> guard(lock);
        ifstream file(filePath.c_str(), ios::binary);
        
        if (!file.is_open()) {
            logFatal(LOG_SEG_IO, "File not found: " + filePath);
        }
        
        file.seekg((sampleID % 10000) * (32 * 32 * 3 + 1) + 1, ios::beg);
        file.read((char *) fileBuffer + tid * 32 * 32 * 3, 32 * 32 * 3);
        file.close();
    }
    
    cv::Mat img(32, 32, CV_8UC3);
    buf2Mat(img, fileBuffer + tid * 32 * 32 * 3);
    return img;
}

void CIFAR_10ReadFunc::readNxtLabel(unsigned int batchSampleID, unsigned int sampleID, Data& data, int tid, int tc, bool istest) {
    int fileID = (int)sampleID / 10000 + 1;
    
    string filePath = this->path + (istest ? "\\test_batch.bin" : "\\data_batch_" + to_string(fileID) + ".bin");
    
    unsigned char label = 0;
    {
        lock_guard<mutex> guard(lock);
        ifstream file(filePath, ios::binary);
        
        if (!file.is_open()) {
            logFatal(LOG_SEG_IO, "File not found: " + filePath);
        }
        
        file.seekg((sampleID % 10000) * (32 * 32 * 3 + 1), ios::beg);
        file.read((char *) &label, 1);
        file.close();
    }
    
    memset(data.Y->data, 0, data.Y->sizes.size * data.Y->elemSize);
    switch (data.Y->dataType) {
        case CUDNN_DATA_HALF:
            ((half*)data.Y->data)[(int)label] = 1;
            break;
        case CUDNN_DATA_FLOAT:
            ((float*)data.Y->data)[(int)label] = 1;
            break;
        case CUDNN_DATA_DOUBLE:
            ((double*)data.Y->data)[(int)label] = 1;
            break;
        default:
            throw runtime_error("Unsupported data type");
    }
}
