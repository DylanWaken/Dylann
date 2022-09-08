//
// Created by Dylan on 9/3/2022.
//

#ifndef DYLANN_INSTRUCTIONS_CUH
#define DYLANN_INSTRUCTIONS_CUH

#include "../ops/cuTensorOps.cuh"
#include "../ops/cuReduce.cuh"
#include "../ops/cuActivation.cuh"
#include "../ops/cuConv.cuh"
#include "../ops/cuLinear.cuh"
#include "../ops/cuBatchnorm.cuh"
#include "../ops/cuPool.cuh"
#include "../ops/cuConcat.cuh"
#include "../ops/cuDropout.cuh"

#define INS_ADD 0
#define INS_SCALE 1
#define INS_LINEAR 2
#define INS_CONV2D 3
#define INS_MAXPOOL2D 4
#define INS_AVGPOOL2D 5
#define INS_SOFTMAX 6
#define INS_BATCHNROM 7
#define INS_SOFTMAX_LOG 8
#define INS_CONCAT_CHANNEL 9
#define INS_DROPOUT 10
#define INS_FLATTEN 11
#define INS_GLOBAL_AVGPOOL 12

#define INS_RELU 100
#define INS_SIGMOID 101
#define INS_TANH 102
#define INS_ELU 103
#define INS_SWISH 104
#define INS_CLIPPED_RELU 105

namespace dylann {
    
    /**
     * Operations are the base units of this engine
     * after the modules are constructed
     * the model will be serialized to these base instructions
     * like assembly instructions
     */
    struct Operation {
    public:
        unsigned int opCode{};
        unsigned int paramCount{};
        
        //control train or inference
        bool train = true;
        map<TENSOR_PTR ,cuTensorBase*>* params{};
        
        Operation(unsigned int opCode, unsigned int paramCount) : opCode(opCode), paramCount(paramCount) {}
        
        void bind( map<TENSOR_PTR ,cuTensorBase*>* pBase) {
            this->params = pBase;
        }
        
        virtual void run() = 0;
    
        virtual size_t getEncodedSize() = 0;
        
        virtual void print() = 0;
        
        virtual void encodeParams(unsigned char * file, size_t &offset) = 0;
    };
    
    struct ADD : public Operation {
    public:
        TENSOR_PTR A;
        TENSOR_PTR B;
        float alpha;
        float beta;
        
        ADD(TENSOR_PTR A, TENSOR_PTR B, float alpha, float beta) :
                Operation(INS_ADD, 4), A(A), B(B), alpha(alpha), beta(beta) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(float) * 2;
        }
        
        void print() override {
            cout << "ADD " << std::hex << A << " " << std::hex
            << B << " " << std::dec << alpha << " " << beta << endl;
        }
    };
    
    struct SCALE : public Operation {
    public:
        TENSOR_PTR A;
        float alpha;
        
        SCALE(TENSOR_PTR A, float alpha) :
                Operation(INS_SCALE, 2), A(A), alpha(alpha){}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) + sizeof(float);
        }
        
        void print() override {
            cout << "SCALE " << std::hex << A << " " << std::dec << alpha << endl;
        }
    };
    
    struct LINEAR : public Operation {
    public:
        TENSOR_PTR W;
        TENSOR_PTR B;
        TENSOR_PTR X;
        TENSOR_PTR Y;
        
        LINEAR(TENSOR_PTR W, TENSOR_PTR B, TENSOR_PTR X, TENSOR_PTR Y) :
                Operation(INS_LINEAR, 4), W(W), B(B), X(X), Y(Y) {}
                
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 4;
        }
        
        void print() override {
            cout << "LINEAR " << std::hex << W << " " << B << " " << X << " " << Y << endl;
        }
    };
    
    struct CONV2D : public Operation {
    public:
        TENSOR_PTR W;
        TENSOR_PTR B;
        TENSOR_PTR X;
        TENSOR_PTR Y;
        int strideH;
        int strideW;
        int padH;
        int padW;
        int dilationH;
        int dilationW;
        
        CONV2D(TENSOR_PTR W, TENSOR_PTR B, TENSOR_PTR X, TENSOR_PTR Y, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW) :
                Operation(INS_CONV2D, 10), W(W), B(B), X(X), Y(Y),
                strideH(strideH), strideW(strideW), padH(padH), padW(padW), dilationH(dilationH), dilationW(dilationW) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 4 + sizeof(int) * 6;
        }
        
        void print() override {
            cout << "CONV2D " << std::hex << W << " " << B << " " << X << " " << Y << " " << std::dec
            << strideH << " " << strideW << " " << padH << " " << padW << " " << dilationH << " " << dilationW << endl;
        }
    };
    
    struct MAXPOOL2D : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        int kernelH;
        int kernelW;
        int strideH;
        int strideW;
        int padH;
        int padW;
        
        MAXPOOL2D(TENSOR_PTR X, TENSOR_PTR Y, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW) :
                Operation(INS_MAXPOOL2D, 8), X(X), Y(Y),
                kernelH(kernelH), kernelW(kernelW), strideH(strideH), strideW(strideW), padH(padH), padW(padW) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(int) * 6;
        }
        
        void print() override {
            cout << "MAXPOOL2D " << std::hex << X << " " << Y << " " << std::dec
            << kernelH << " " << kernelW << " " << strideH << " " << strideW << " " << padH << " " << padW << endl;
        }
    };
    
    struct AVGPOOL2D : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        int kernelH;
        int kernelW;
        int strideH;
        int strideW;
        int padH;
        int padW;
        
        AVGPOOL2D(TENSOR_PTR X, TENSOR_PTR Y, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW) :
                Operation(INS_AVGPOOL2D, 8), X(X), Y(Y),
                kernelH(kernelH), kernelW(kernelW), strideH(strideH), strideW(strideW), padH(padH), padW(padW) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(int) * 6;
        }
        
        void print() override {
            cout << "AVGPOOL2D " << std::hex << X << " " << Y << " " << std::dec
            << kernelH << " " << kernelW << " " << strideH << " " << strideW << " " << padH << " " << padW << endl;
        }
    };
    
    struct SOFTMAX : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        int step;
        
        SOFTMAX(TENSOR_PTR X, TENSOR_PTR Y, int step) :
                Operation(INS_SOFTMAX, 3), X(X), Y(Y), step(step) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2;
        }
        
        void print() override {
            cout << "SOFTMAX " << std::hex << X << " " << Y << " " << std::dec << step << endl;
        }
    };
    
    struct BATCHNROM : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        TENSOR_PTR gamma;
        TENSOR_PTR beta;
        TENSOR_PTR mean;
        TENSOR_PTR var;
        float eps;
        float expAvgFactor;
        
        BATCHNROM(TENSOR_PTR X, TENSOR_PTR Y, TENSOR_PTR gamma, TENSOR_PTR beta, TENSOR_PTR mean,
                  TENSOR_PTR var, float eps, float expAvgFactor) :
                Operation(INS_BATCHNROM, 8), X(X), Y(Y), gamma(gamma), beta(beta),
                mean(mean), var(var), eps(eps), expAvgFactor(expAvgFactor) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 6 + sizeof(float) * 2;
        }
        
        void print() override {
            cout << "BATCHNROM " << std::hex << X << " " << Y << " " << gamma << " " << beta << " " << mean << " " << var << " " << std::dec
            << eps << " " << expAvgFactor << endl;
        }
    };
    
    struct SOFTMAX_LOG : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        int step;
        
        SOFTMAX_LOG(TENSOR_PTR X, TENSOR_PTR Y, int step) :
                Operation(INS_SOFTMAX_LOG, 3), X(X), Y(Y), step(step) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2;
        }
        
        void print() override {
            cout << "SOFTMAX_LOG " << std::hex << X << " " << Y << " " << std::dec << step << endl;
        }
    };
    
    struct CONCAT_CHANNEL : public Operation {
    public:
        TENSOR_PTR* X{};
        TENSOR_PTR Y;
        int paramC;
        
        CONCAT_CHANNEL(std::initializer_list<TENSOR_PTR> X, TENSOR_PTR Y) :
                Operation(INS_CONCAT_CHANNEL, 2), Y(Y), paramC((int)X.size()) {
            
            cudaMallocHost(&this->X, sizeof(TENSOR_PTR) * paramCount);
            for (auto i = 0 ; i < paramC; i++) {
                 this->X[i] = X.begin()[i];
                 paramCount++;
            }
        }
        
        CONCAT_CHANNEL(TENSOR_PTR* X, TENSOR_PTR Y, int paramC) :
                Operation(INS_CONCAT_CHANNEL, 2), X(X), Y(Y), paramC(paramC) {
            paramCount += paramC;
        }
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        void print() override {
            cout << "CONCAT_CHANNEL ";
            for (auto i = 0 ; i < paramC; i++) {
                cout << std::hex << X[i] << " ";
            }
            cout << std::hex << Y << " " << std::dec << paramC << endl;
        }
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * (paramC + 1);
        }
    };
    
    struct DROPOUT : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        float rate;
        
        DROPOUT(TENSOR_PTR X, TENSOR_PTR Y, float rate) :
                Operation(INS_DROPOUT, 3), X(X), Y(Y), rate(rate) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(float);
        }
        
        void print() override {
            cout << "DROPOUT " << std::hex << X << " " << Y << " " << std::dec << rate << endl;
        }
    };
    
    struct FLATTEN : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        
        FLATTEN(TENSOR_PTR X, TENSOR_PTR Y) :
                Operation(INS_FLATTEN, 2), X(X), Y(Y) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2;
        }
        
        void print() override {
            cout << "FLATTEN " << std::hex << X << " " << Y << endl;
        }
    };
    
    
    //Activations --------------------------------------------------------
    
    struct RELU : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        
        RELU(TENSOR_PTR X, TENSOR_PTR Y) :
                Operation(INS_RELU, 2), X(X), Y(Y) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2;
        }
        
        void print() override {
            cout << "RELU " << std::hex << X << " " << Y << endl;
        }
    };
    
    struct SIGMOID : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        
        SIGMOID(TENSOR_PTR X, TENSOR_PTR Y) :
                Operation(INS_SIGMOID, 2), X(X), Y(Y) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2;
        }
        
        void print() override {
            cout << "SIGMOID " << std::hex << X << " " << Y << endl;
        }
    };
    
    struct TANH : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        
        TANH(TENSOR_PTR X, TENSOR_PTR Y) :
                Operation(INS_TANH, 2), X(X), Y(Y) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2;
        }
        
        void print() override {
            cout << "TANH " << std::hex << X << " " << Y << endl;
        }
    };
    
    struct ELU : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        float alpha;
        
        ELU(TENSOR_PTR X, TENSOR_PTR Y, float alpha) :
                Operation(INS_ELU, 3), X(X), Y(Y), alpha(alpha) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(float);
        }
        
        void print() override {
            cout << "ELU " << std::hex << X << " " << Y << " " << std::dec << alpha << endl;
        }
    };
    
    struct SWISH : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        float beta;
        
        SWISH(TENSOR_PTR X, TENSOR_PTR Y, float beta) :
                Operation(INS_SWISH, 3), X(X), Y(Y), beta(beta) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(float);
        }
        
        void print() override {
            cout << "SWISH " << std::hex << X << " " << Y << " " << std::dec << beta << endl;
        }
    };
    
    struct CLIPPED_RELU : public Operation {
    public:
        TENSOR_PTR X;
        TENSOR_PTR Y;
        float threshold;
        
        CLIPPED_RELU(TENSOR_PTR X, TENSOR_PTR Y, float threshold) :
                Operation(INS_CLIPPED_RELU, 3), X(X), Y(Y), threshold(threshold) {}
                
        void run() override;
        
        void encodeParams(unsigned char * file, size_t &offset) override;
        
        size_t getEncodedSize() override {
            return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(float);
        }
        
        void print() override {
            cout << "CLIPPED_RELU " << std::hex << X << " " << Y << " " << std::dec << threshold << endl;
        }
    };
    
    //EXTRACT
    ADD* extractAdd(const unsigned char * file, size_t &offset);
    
    SCALE* extractScale(const unsigned char * file, size_t &offset);
    
    LINEAR* extractLinear(const unsigned char * file, size_t &offset);
    
    CONV2D* extractConv2D(const unsigned char * file, size_t &offset);
    
    MAXPOOL2D* extractMaxPool2D(const unsigned char * file, size_t &offset);
    
    AVGPOOL2D* extractAvgPool2D(const unsigned char * file, size_t &offset);
    
    SOFTMAX* extractSoftmax(const unsigned char * file, size_t &offset);
    
    BATCHNROM* extractBatchNorm(const unsigned char * file, size_t &offset);
    
    SOFTMAX_LOG* extractSoftmaxLog(const unsigned char * file, size_t &offset);
    
    CONCAT_CHANNEL* extractConcatChannel(const unsigned char * file, size_t &offset);
    
    DROPOUT* extractDropout(const unsigned char * file, size_t &offset);
    
    //Extract Activations
    RELU* extractRelu(const unsigned char * file, size_t &offset);
    
    SIGMOID* extractSigmoid(const unsigned char * file, size_t &offset);
    
    TANH* extractTanh(const unsigned char * file, size_t &offset);
    
    ELU* extractElu(const unsigned char * file, size_t &offset);
    
    SWISH* extractSwish(const unsigned char * file, size_t &offset);
    
    CLIPPED_RELU* extractClippedRelu(const unsigned char * file, size_t &offset);
    
} // dylann

#endif //DYLANN_INSTRUCTIONS_CUH
