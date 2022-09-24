#include <iostream>

#include "presets/Resnet.cuh"
#include "dylann/module/Sequence.cuh"

#include <thread>
#include <mutex>
#include <condition_variable>
#include "io/dataset/DataPipeline.cuh"
#include "cudautil/ThreadController.cuh"


using namespace dylann;
using namespace std;
using namespace io;

struct TestReadFunc : public ReadFuncCV{
    cv::Mat readNxt(unsigned int sampleID) override {
        this_thread::sleep_for(chrono::milliseconds(1));
        return cv::Mat(224, 224, CV_8UC3, cv::Scalar(sampleID%256, sampleID%256, sampleID%256));
    }
    
    void readNxtLabel(unsigned int seq, unsigned int sampleID, Data* data) override {
        this_thread::sleep_for(chrono::milliseconds(1));
        ((float*)data->Y->data)[0] = sampleID;
    }
};

struct testAug : public AugmentIns{
    cv::Mat augment(cv::Mat& input) override {
        this_thread::sleep_for(chrono::milliseconds(1));
        return input;
    }
};

int main() {
//    initEngineContext();
//
//    auto X = cuTensor::create<CUDNN_DATA_FLOAT>(shape4(64, 64, 32, 32), 0);
//
//    ResnetConv id = ResnetConv();
//    auto Y = id.forward(X);
//    auto Z = id.forward(Y);
//    auto W = id.forward(Z);
//
//    auto seq = ctx2seq();
//    seq->setOpt(new Momentum(0.001));
//    seq->randomizeParams();
//
//    for (int i = 0; i < 1; i++) {
//        seq->forward();
//        seq->backward();
//        W.print();
//        seq->opt->apply();
//    }

    

    Data* dataset;
    cudaMallocHost(&dataset, 10000 * sizeof(Data));
    for (int i = 0; i < 10000; i++){
        dataset[i].X = HostTensorBase::create(shape4(224, 224, 3, 1), CUDNN_DATA_FLOAT);
        dataset[i].Y = HostTensorBase::create(shape4(1, 1, 1, 1), CUDNN_DATA_FLOAT);
    }
    
    cout<<"start reading"<<endl;
    vector<AugmentIns*> augIns;
    augIns.push_back(new testAug());
    
    auto* readf = new TestReadFunc();
    
    _alloc<16>(pipelineCV, readf, ref(augIns), dataset, 10000, 0);
}
