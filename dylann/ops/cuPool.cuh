//
// Created by Dylan on 9/4/2022.
//

#ifndef DYLANN_CUPOOL_CUH
#define DYLANN_CUPOOL_CUH

#include "cuTensorOps.cuh"
#include "cuTensorOpGrads.cuh"

namespace dylann {
    cuTensorBase *maxPoolOp(cuTensorBase* X, cuTensorBase* Y, int rangeH, int rangeW,
                            int padH, int padW, int strideH, int strideW);
    
    cuTensorBase *maxPoolOpGrads(cuTensorBase* X, cuTensorBase* Y,  int rangeH, int rangeW,
                                 int padH, int padW, int strideH, int strideW);
    
    
    cuTensorBase *avgPoolOp(cuTensorBase* X, cuTensorBase* Y, int rangeH, int rangeW,
                            int padH, int padW, int strideH, int strideW);
    
    cuTensorBase *avgPoolOpGrads(cuTensorBase* X, cuTensorBase* Y,  int rangeH, int rangeW,
                                 int padH, int padW, int strideH, int strideW);
    
    
    struct GRAD_MAXPOOL : public GradTracker{
        cuTensorBase* X;
        int rangeH, rangeW, padH, padW, strideH, strideW;
        explicit GRAD_MAXPOOL(cuTensorBase* X,  int rangeH, int rangeW,
                              int padH, int padW, int strideH, int strideW) : X(X),
                            rangeH(rangeH), rangeW(rangeW),
                            padH(padH), padW(padW),
                            strideH(strideH), strideW(strideW){}
        
        void backwardCalc(cuTensorBase* Y) override;
    };
    
    struct GRAD_AVGPOOL : public GradTracker{
        cuTensorBase* X;
        int rangeH, rangeW, padH, padW, strideH, strideW;
        explicit GRAD_AVGPOOL(cuTensorBase* X,  int rangeH, int rangeW,
                              int padH, int padW, int strideH, int strideW) : X(X),
                            rangeH(rangeH), rangeW(rangeW),
                            padH(padH), padW(padW),
                            strideH(strideH), strideW(strideW){}
        
        void backwardCalc(cuTensorBase* Y) override;
    };
} // dylann

#endif //DYLANN_CUPOOL_CUH
