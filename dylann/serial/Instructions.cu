//
// Created by Dylan on 9/3/2022.
//

#include "Instructions.cuh"

namespace dylann {
    void ADD::run() {
        addOp(params[A], params[B], alpha, beta);
    }
    
    void ADD::encodeParams(unsigned char *file, unsigned int &offset) {
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        *(TENSOR_PTR*)(file + offset) = A;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = B;
        offset += sizeof(TENSOR_PTR);
        *(float*)(file + offset) = alpha;
        offset += sizeof(float);
        *(float*)(file + offset) = beta;
        offset += sizeof(float);
    }
    
    void SCALE::run() {
        scale(params[A], alpha);
    }
    
    void SCALE::encodeParams(unsigned char *file, unsigned int &offset) {
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        *(TENSOR_PTR*)(file + offset) = A;
        offset += sizeof(TENSOR_PTR);
        *(float*)(file + offset) = alpha;
        offset += sizeof(float);
    }
    
    void LINEAR::run() {
        linearOp(params[W], params[B], params[X], params[Y]);
    }
    
    void LINEAR::encodeParams(unsigned char *file, unsigned int &offset) {
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        *(TENSOR_PTR*)(file + offset) = W;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = B;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
    }
    
    void CONV2D::run() {
        conv2dOp(params[W], params[B], params[X], params[Y], padH, padW, strideH, strideW, dilationH, dilationW);
    }
    
    void CONV2D::encodeParams(unsigned char *file, unsigned int &offset) {
        *(unsigned int*)(file + offset) = opCode;
        offset += sizeof(unsigned int);
        *(unsigned int*)(file + offset) = paramCount;
        offset += sizeof(unsigned int);
        
        *(TENSOR_PTR*)(file + offset) = W;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = B;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = X;
        offset += sizeof(TENSOR_PTR);
        *(TENSOR_PTR*)(file + offset) = Y;
        offset += sizeof(TENSOR_PTR);
        *(int*)(file + offset) = strideH;
        offset += sizeof(int);
        *(int*)(file + offset) = strideW;
        offset += sizeof(int);
        *(int*)(file + offset) = padH;
        offset += sizeof(int);
        *(int*)(file + offset) = padW;
        offset += sizeof(int);
        *(int*)(file + offset) = dilationH;
        offset += sizeof(int);
        *(int*)(file + offset) = dilationW;
        offset += sizeof(int);
    }
    
    //EXTRACT
    //------------------------------------------------------
    
    ADD* extractAdd(const unsigned char * file, unsigned int & offset) {
        TENSOR_PTR A = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR B = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float alpha = *(float*)(file + offset);
        offset += sizeof(float);
        float beta = *(float*)(file + offset);
        offset += sizeof(float);
        
        auto* add = new ADD(A, B, alpha, beta);
        return add;
    }
    
    SCALE* extractScale(const unsigned char * file, unsigned int & offset) {
        TENSOR_PTR A = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        float alpha = *(float*)(file + offset);
        offset += sizeof(float);
        
        auto* scale = new SCALE(A, alpha);
        return scale;
    }
    
    LINEAR* extractLinear(const unsigned char * file, unsigned int & offset) {
        TENSOR_PTR W = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR B = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        
        auto* linear = new LINEAR(W, B, X, Y);
        return linear;
    }
    
    CONV2D* extractConv2D(const unsigned char * file, unsigned int & offset){
        TENSOR_PTR W = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR B = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR X = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        TENSOR_PTR Y = *(TENSOR_PTR*)(file + offset);
        offset += sizeof(TENSOR_PTR);
        int strideH = *(int*)(file + offset);
        offset += sizeof(int);
        int strideW = *(int*)(file + offset);
        offset += sizeof(int);
        int padH = *(int*)(file + offset);
        offset += sizeof(int);
        int padW = *(int*)(file + offset);
        offset += sizeof(int);
        int dilationH = *(int*)(file + offset);
        offset += sizeof(int);
        int dilationW = *(int*)(file + offset);
        offset += sizeof(int);
        
        auto* conv2d = new CONV2D(W, B, X, Y, strideH, strideW, padH, padW, dilationH, dilationW);
        return conv2d;
    }
} // dylann