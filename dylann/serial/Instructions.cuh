//
// Created by Dylan on 9/3/2022.
//

#ifndef DYLANN_INSTRUCTIONS_CUH
#define DYLANN_INSTRUCTIONS_CUH

#include "../ops/cuTensorOps.cuh"
#include "../ops/cuTensorOpGrads.cuh"
#include "../ops/cuReduce.cuh"
#include "../ops/cuActivation.cuh"
#include "../ops/cuConv.cuh"
#include "../ops/cuLinear.cuh"

#define INS_ADD 0
#define INS_SCALE 1
#define INS_LINEAR 2

namespace dylann {
    typedef unsigned int TENSOR_PTR;
    
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
        cuTensorBase** params{};
        
        Operation(unsigned int opCode, unsigned int paramCount) : opCode(opCode), paramCount(paramCount) {}
        
        void bind(cuTensorBase** pBase) {
            this->params = pBase;
        }
        
        virtual void run() = 0;
        
        virtual void encodeParams(unsigned char * file, unsigned int & offset) = 0;
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
        
        void encodeParams(unsigned char * file, unsigned int & offset) override;
    };
    
    struct SCALE : public Operation {
    public:
        TENSOR_PTR A;
        float alpha;
        
        SCALE(TENSOR_PTR A, float alpha) :
                Operation(INS_SCALE, 2), A(A), alpha(alpha){}
        
        void run() override;
        
        void encodeParams(unsigned char * file, unsigned int & offset) override;
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
        
        void encodeParams(unsigned char * file, unsigned int & offset) override;
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
                Operation(INS_LINEAR, 10), W(W), B(B), X(X), Y(Y),
                strideH(strideH), strideW(strideW), padH(padH), padW(padW), dilationH(dilationH), dilationW(dilationW) {}
        
        void run() override;
        
        void encodeParams(unsigned char * file, unsigned int & offset) override;
    };
    
    
    
    //EXTRACT
    
    ADD* extractAdd(const unsigned char * file, unsigned int & offset);
    
    SCALE* extractScale(const unsigned char * file, unsigned int & offset);
    
    LINEAR* extractLinear(const unsigned char * file, unsigned int & offset);
    
    CONV2D* extractConv2D(const unsigned char * file, unsigned int & offset);
    
} // dylann

#endif //DYLANN_INSTRUCTIONS_CUH
