//
// Created by Dylan on 9/7/2022.
//

#ifndef DYLANN_GRADINSTRUCTIONS_CUH
#define DYLANN_GRADINSTRUCTIONS_CUH

#include "Instructions.cuh"

#define INS_GRADS_ADD 1000
#define INS_GRADS_SCALE 1001
#define INS_GRADS_LINEAR 1002
#define INS_GRADS_CONV2D 1003
#define INS_GRADS_MAXPOOL2D 1004
#define INS_GRADS_AVGPOOL2D 1005
#define INS_GRADS_SOFTMAX 1006
#define INS_GRADS_BATCHNORM 1007
#define INS_GRADS_SOFTMAX_LOG 1008
#define INS_GRADS_CONCAT_CHANNEL 1009
#define INS_GRADS_DROPOUT 1010
#define INS_GRADS_FLATTEN 1011
#define INS_GRADS_GLOBAL_AVGPOOL 1012
#define INS_GRADS_SOFTMAX_CE 1013

#define INS_GRADS_RELU 1100
#define INS_GRADS_SIGMOID 1101
#define INS_GRADS_TANH 1102
#define INS_GRADS_ELU 1103
#define INS_GRADS_SWISH 1104
#define INS_GRADS_CLIPPED_RELU 1105


namespace dylann {
     
        struct ADD_GRADS : public Operation{
            public:
            TENSOR_PTR A;
            TENSOR_PTR B;
            float alpha;
            float beta;
            
            ADD_GRADS(TENSOR_PTR A, TENSOR_PTR B, float alpha, float beta)
                 : Operation(INS_GRADS_ADD, 4), A(A), B(B), alpha(alpha), beta(beta) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                 return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(float) * 2;
            }
            
            void print() override {
                 cout << "GRADS_ADD 0x" << std::hex << A << " 0x" << std::hex
                      << B << " " << std::dec << alpha << " " << beta << endl;
            }
        };
        
        struct SCALE_GRADS : public Operation{
            public:
            TENSOR_PTR A;
            float alpha;
            
            SCALE_GRADS(TENSOR_PTR A, float alpha)
                 : Operation(INS_GRADS_SCALE, 2), A(A), alpha(alpha) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                 return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) + sizeof(float);
            }
            
            void print() override {
                 cout << "GRADS_SCALE 0x" << std::hex << A << " " << std::dec << alpha << endl;
            }
        };
     
        struct LINEAR_GRADS : public Operation{
            public:
            TENSOR_PTR W;
            TENSOR_PTR B;
            TENSOR_PTR X;
            TENSOR_PTR Y;
            
            LINEAR_GRADS(TENSOR_PTR W, TENSOR_PTR B, TENSOR_PTR X, TENSOR_PTR Y)
            : Operation(INS_GRADS_LINEAR, 4), W(W), B(B), X(X), Y(Y) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                  return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 4;
            }
            
            void print() override {
                  cout << "GRADS_LINEAR 0x" << std::hex << W << " 0x" << B << " 0x" << X << " 0x" << Y << endl;
            }
        };
        
        struct CONV2D_GRADS : public Operation{
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
            
            CONV2D_GRADS(TENSOR_PTR W, TENSOR_PTR B, TENSOR_PTR X, TENSOR_PTR Y, int strideH, int strideW,
                         int padH, int padW, int dilationH, int dilationW) :
            Operation(INS_GRADS_CONV2D, 10), W(W), B(B), X(X), Y(Y), strideH(strideH),
                        strideW(strideW), padH(padH), padW(padW), dilationH(dilationH), dilationW(dilationW) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 4 + sizeof(int) * 6;
            }
            
            void print() override {
                cout << "GRADS_CONV2D 0x" << std::hex << W << " 0x" << B << " 0x" << X << " 0x" << Y << " " <<
                std::dec << strideH << " " << strideW << " " << padH << " " << padW << " " << dilationH << " " << dilationW << endl;
            }
        };
        
        struct MAXPOOL2D_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            int kernelH;
            int kernelW;
            int strideH;
            int strideW;
            int padH;
            int padW;
            
            MAXPOOL2D_GRADS(TENSOR_PTR X, TENSOR_PTR Y, int kernelH, int kernelW, int strideH, int strideW,
                            int padH, int padW) :
            Operation(INS_GRADS_MAXPOOL2D, 8), X(X), Y(Y), kernelH(kernelH), kernelW(kernelW),
                    strideH(strideH), strideW(strideW), padH(padH), padW(padW) {}
                    
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(int) * 6;
            }
            
            void print() override {
                cout << "GRADS_MAXPOOL2D 0x" << std::hex << X << " 0x" << Y << " " <<
                std::dec << kernelH << " " << kernelW << " " << strideH << " " << strideW << " " << padH << " " << padW << endl;
            }
        };
        
        struct AVGPOOL2D_GRADS : public Operation{
            public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            int kernelH;
            int kernelW;
            int strideH;
            int strideW;
            int padH;
            int padW;
            
            AVGPOOL2D_GRADS(TENSOR_PTR X, TENSOR_PTR Y, int kernelH, int kernelW, int strideH, int strideW,
                            int padH, int padW) :
            Operation(INS_GRADS_AVGPOOL2D, 8), X(X), Y(Y), kernelH(kernelH), kernelW(kernelW),
                    strideH(strideH), strideW(strideW), padH(padH), padW(padW) {}
                    
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(int) * 6;
            }
            
            void print() override {
                cout << "GRADS_AVGPOOL2D 0x" << std::hex << X << " 0x" << Y << " " <<
                std::dec << kernelH << " " << kernelW << " " << strideH << " " << strideW << " " << padH << " " << padW << endl;
            }
        };
        
        struct SOFTMAX_GRADS : public Operation{
            public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            int step;
    
            SOFTMAX_GRADS(TENSOR_PTR X, TENSOR_PTR Y, int step) :
            Operation(INS_GRADS_SOFTMAX, 3), X(X), Y(Y), step(step) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(int);
            }
            
            void print() override {
                cout << "GRADS_SOFTMAX 0x" << std::hex << X << " 0x" << Y << " " << std::dec << step << endl;
            }
        };
        
        struct BATCHNORM_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            TENSOR_PTR gamma;
            TENSOR_PTR beta;
            TENSOR_PTR mean;
            TENSOR_PTR var;
            float eps;
            float expAvgFactor;
    
            BATCHNORM_GRADS(TENSOR_PTR X, TENSOR_PTR Y, TENSOR_PTR gamma, TENSOR_PTR beta, TENSOR_PTR mean,
                            TENSOR_PTR var, float eps, float expAvgFactor) :
            Operation(INS_GRADS_BATCHNORM, 8), X(X), Y(Y), gamma(gamma), beta(beta),
                            mean(mean), var(var), eps(eps), expAvgFactor(expAvgFactor) {}
                            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 6 + sizeof(float) * 2;
            }
            
            void print() override {
                cout << "GRADS_BATCHNORM 0x" << std::hex << X << " 0x" << Y << " 0x" << gamma << " 0x" << beta << " 0x" << mean << " 0x" << var << " " <<
                std::dec << eps << " " << expAvgFactor << endl;
            }
        };
        
        struct SOFTMAX_LOG_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            int step;
    
            SOFTMAX_LOG_GRADS(TENSOR_PTR X, TENSOR_PTR Y, int step) :
            Operation(INS_GRADS_SOFTMAX_LOG, 3), X(X), Y(Y), step(step) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(int);
            }
            
            void print() override {
                cout << "GRADS_SOFTMAX_LOG 0x" << std::hex << X << " 0x" << Y << " " << std::dec << step << endl;
            }
        };
        
        struct CONCAT_CHANNEL_GRADS : public Operation{
            TENSOR_PTR* X{};
            TENSOR_PTR Y;
            int paramC;
            
            CONCAT_CHANNEL_GRADS(TENSOR_PTR* X, TENSOR_PTR Y, int paramC) :
            Operation(INS_GRADS_CONCAT_CHANNEL, 3), X(X), Y(Y), paramC(paramC) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * (paramC + 1) + sizeof(int);
            }
            
            void print() override {
                cout << "GRADS_CONCAT_CHANNEL 0x" << std::hex ;
                for(int i = 0; i < paramC; i++){
                    cout << X[i] << " 0x";
                }
                cout <<  Y << std::dec << " " << paramC << " " << endl;
                cout << std::dec << endl;
            }
        };
        
        struct DROPOUT_GRADS : public Operation{
            TENSOR_PTR X;
            TENSOR_PTR Y;
            float p;
            
            DROPOUT_GRADS(TENSOR_PTR X, TENSOR_PTR Y, float p) :
            Operation(INS_GRADS_DROPOUT, 4), X(X), Y(Y), p(p) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 3 + sizeof(float);
            }
            
            void print() override {
                cout << "GRADS_DROPOUT 0x" << std::hex << X << " 0x" << Y <<  " " << std::dec << p << endl;
            }
        };
        
        struct FLATTEN_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            
            FLATTEN_GRADS(TENSOR_PTR X, TENSOR_PTR Y) :
            Operation(INS_GRADS_FLATTEN, 2), X(X), Y(Y) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2;
            }
            
            void print() override {
                cout << "GRADS_FLATTEN 0x" << std::hex << X << " 0x" << Y << std::dec << endl;
            }
        };
        
        struct GLOBAL_AVGPOOL_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            
            GLOBAL_AVGPOOL_GRADS(TENSOR_PTR X, TENSOR_PTR Y) :
            Operation(INS_GRADS_GLOBAL_AVGPOOL, 2), X(X), Y(Y) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2;
            }
            
            void print() override {
                cout << "GRADS_GLOBAL_AVGPOOL 0x" << std::hex << X << " 0x" << Y << std::dec << endl;
            }
        };
        
        struct SOFTMAX_CE_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            int step;
    
            SOFTMAX_CE_GRADS(TENSOR_PTR X, TENSOR_PTR Y, int step) :
            Operation(INS_GRADS_SOFTMAX_CE, 3), X(X), Y(Y), step(step) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(int);
            }
            
            void print() override {
                cout << "GRADS_SOFTMAX_CE 0x" << std::hex << X << " 0x" << Y << " " << std::dec << step << endl;
            }
        };
        
        //Activations ==========================
        
        struct RELU_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            
            RELU_GRADS(TENSOR_PTR X, TENSOR_PTR Y) :
            Operation(INS_GRADS_RELU, 2), X(X), Y(Y) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2;
            }
            
            void print() override {
                cout << "GRADS_RELU 0x" << std::hex << X << " 0x" << Y << std::dec << endl;
            }
        };
        
        struct SIGMOID_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            
            SIGMOID_GRADS(TENSOR_PTR X, TENSOR_PTR Y) :
            Operation(INS_GRADS_SIGMOID, 2), X(X), Y(Y) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2;
            }
            
            void print() override {
                cout << "GRADS_SIGMOID 0x" << std::hex << X << " 0x" << Y << std::dec << endl;
            }
        };
        
        struct TANH_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            
            TANH_GRADS(TENSOR_PTR X, TENSOR_PTR Y) :
            Operation(INS_GRADS_TANH, 2), X(X), Y(Y) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2;
            }
            
            void print() override {
                cout << "GRADS_TANH 0x" << std::hex << X << " 0x" << Y << std::dec << endl;
            }
        };
        
        struct ELU_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            float alpha;
            
            ELU_GRADS(TENSOR_PTR X, TENSOR_PTR Y, float alpha) :
            Operation(INS_GRADS_ELU, 3), X(X), Y(Y), alpha(alpha) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(float);
            }
            
            void print() override {
                cout << "GRADS_ELU 0x" << std::hex << X << " 0x" << Y << " " << std::dec << alpha << endl;
            }
        };
        
        struct SWISH_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            float beta;
            
            SWISH_GRADS(TENSOR_PTR X, TENSOR_PTR Y, float beta) :
            Operation(INS_GRADS_SWISH, 3), X(X), Y(Y), beta(beta) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(float);
            }
            
            void print() override {
                cout << "GRADS_SWISH 0x" << std::hex << X << " 0x" << Y << " " << std::dec << beta << endl;
            }
        };
        
        struct CLIPPED_RELU_GRADS : public Operation{
        public:
            TENSOR_PTR X;
            TENSOR_PTR Y;
            float threshold;
            
            CLIPPED_RELU_GRADS(TENSOR_PTR X, TENSOR_PTR Y, float threshold) :
            Operation(INS_GRADS_CLIPPED_RELU, 3), X(X), Y(Y), threshold(threshold) {}
            
            void run() override;
            
            void encodeParams(unsigned char * file, size_t &offset) override;
            
            size_t getEncodedSize() override {
                return sizeof(unsigned int) * 2 + sizeof(TENSOR_PTR) * 2 + sizeof(float);
            }
            
            void print() override {
                cout << "GRADS_CLIPPED_RELU 0x" << std::hex << X << " 0x" << Y << " " << std::dec << threshold << endl;
            }
        };

        //EXTRACT ----------------------------------------------------
     
        ADD_GRADS* extractAddGrads(const unsigned char * file, size_t &offset);
     
        SCALE_GRADS* extractScaleGrads(const unsigned char * file, size_t &offset);
        
        LINEAR_GRADS* extractLinearGrads(const unsigned char * file, size_t &offset);
        
        CONV2D_GRADS* extractConv2DGrads(const unsigned char * file, size_t &offset);
        
        MAXPOOL2D_GRADS* extractMaxPool2DGrads(const unsigned char * file, size_t &offset);
        
        AVGPOOL2D_GRADS* extractAvgPool2DGrads(const unsigned char * file, size_t &offset);
        
        SOFTMAX_GRADS* extractSoftmaxGrads(const unsigned char * file, size_t &offset);
        
        BATCHNORM_GRADS* extractBatchNormGrads(const unsigned char * file, size_t &offset);
        
        SOFTMAX_LOG_GRADS* extractSoftmaxLogGrads(const unsigned char * file, size_t &offset);
        
        CONCAT_CHANNEL_GRADS* extractConcatChannelGrads(const unsigned char * file, size_t &offset);
        
        DROPOUT_GRADS* extractDropoutGrads(const unsigned char * file, size_t &offset);
        
        FLATTEN_GRADS* extractFlattenGrads(const unsigned char * file, size_t &offset);
        
        
        RELU_GRADS* extractReluGrads(const unsigned char * file, size_t &offset);
        
        SIGMOID_GRADS* extractSigmoidGrads(const unsigned char * file, size_t &offset);
        
        TANH_GRADS* extractTanhGrads(const unsigned char * file, size_t &offset);
        
        ELU_GRADS* extractEluGrads(const unsigned char * file, size_t &offset);
        
        SWISH_GRADS* extractSwishGrads(const unsigned char * file, size_t &offset);
        
        CLIPPED_RELU_GRADS* extractClippedReluGrads(const unsigned char * file, size_t &offset);
} // dylann

#endif //DYLANN_GRADINSTRUCTIONS_CUH
