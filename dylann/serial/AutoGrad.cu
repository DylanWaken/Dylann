//
// Created by Dylan on 9/14/2022.
//

#include "AutoGrad.cuh"

namespace dylann {
    
    /*
     * Register all forward - backward mapping here
     */
    Operation* getCorrespondBcwd(Operation* input){
        switch (input->opCode) {
            case INS_ADD:
                return new ADD_GRADS(((ADD*)input)->X1, ((ADD*)input)->X2, ((ADD*)input)->Y, ((ADD*)input)->alpha, ((ADD*)input)->beta);
                
            case INS_SCALE:
                return new SCALE_GRADS(((SCALE*)input)->X, ((SCALE*)input)->Y, ((SCALE*)input)->alpha);
                
            case INS_LINEAR:
                return new LINEAR_GRADS(((LINEAR*)input)->W, ((LINEAR*)input)->B, ((LINEAR*)input)->X, ((LINEAR*)input)->Y,
                                        1, 1);
                
            case INS_CONV2D:
                return new CONV2D_GRADS(((CONV2D*)input)->W, ((CONV2D*)input)->B, ((CONV2D*)input)->X, ((CONV2D*)input)->Y,
                                        ((CONV2D*)input)->strideH, ((CONV2D*)input)->strideW, ((CONV2D*)input)->padH, ((CONV2D*)input)->padW,
                                        ((CONV2D*)input)->dilationH, ((CONV2D*)input)->dilationW, 1, 1);
                
            case INS_MAXPOOL2D:
                return new MAXPOOL2D_GRADS(((MAXPOOL2D*)input)->X, ((MAXPOOL2D*)input)->Y, ((MAXPOOL2D*)input)->kernelH, ((MAXPOOL2D*)input)->kernelW,
                                           ((MAXPOOL2D*)input)->strideH, ((MAXPOOL2D*)input)->strideW, ((MAXPOOL2D*)input)->padH, ((MAXPOOL2D*)input)->padW,
                                           1, 1);
                
            case INS_AVGPOOL2D:
                return new AVGPOOL2D_GRADS(((AVGPOOL2D*)input)->X, ((AVGPOOL2D*)input)->Y, ((AVGPOOL2D*)input)->kernelH, ((AVGPOOL2D*)input)->kernelW,
                                           ((AVGPOOL2D*)input)->strideH, ((AVGPOOL2D*)input)->strideW, ((AVGPOOL2D*)input)->padH, ((AVGPOOL2D*)input)->padW,
                                           1, 1);
                
            case INS_GLOBAL_AVGPOOL:
                return new GLOBAL_AVGPOOL_GRADS(((GLOBAL_AVGPOOL2D*)input)->X, ((GLOBAL_AVGPOOL2D*)input)->Y, 1, 1);
                
            case INS_BATCHNORM:
                return new BATCHNORM_GRADS(((BATCHNORM*)input)->X, ((BATCHNORM*)input)->Y, ((BATCHNORM*)input)->gamma, ((BATCHNORM*)input)->beta,
                                           ((BATCHNORM*)input)->mean, ((BATCHNORM*)input)->var, ((BATCHNORM*)input)->eps, ((BATCHNORM*)input)->expAvgFactor,
                                           1, 1);
                
            case INS_BATCHNORM2D:
                return new BATCHNORM2D_GRADS(((BATCHNORM2D*)input)->X, ((BATCHNORM2D*)input)->Y, ((BATCHNORM2D*)input)->gamma, ((BATCHNORM2D*)input)->beta,
                                             ((BATCHNORM2D*)input)->mean, ((BATCHNORM2D*)input)->var, ((BATCHNORM2D*)input)->eps, ((BATCHNORM2D*)input)->expAvgFactor,
                                             1, 1);
                
            case INS_DROPOUT:
                return new DROPOUT_GRADS(((DROPOUT*)input)->X, ((DROPOUT*)input)->Y, ((DROPOUT*)input)->mask,((DROPOUT*)input)->rate);
                
            case INS_RELU:
                return new RELU_GRADS(((RELU*)input)->X, ((RELU*)input)->Y, 1, 1);
                
            case INS_CONCAT_CHANNEL:
                return new CONCAT_CHANNEL_GRADS(((CONCAT_CHANNEL*)input)->X, ((CONCAT_CHANNEL*)input)->Y, ((CONCAT_CHANNEL*)input)->paramC);
                
            case INS_SIGMOID:
                return new SIGMOID_GRADS(((SIGMOID*)input)->X, ((SIGMOID*)input)->Y, 1, 1);
                
            case INS_TANH:
                return new TANH_GRADS(((TANH*)input)->X, ((TANH*)input)->Y, 1, 1);
                
            case INS_ELU:
                return new ELU_GRADS(((ELU*)input)->X, ((ELU*)input)->Y, ((ELU*)input)->alpha, 1, 1);
                
            case INS_SOFTMAX:
                return new SOFTMAX_GRADS(((SOFTMAX*)input)->X, ((SOFTMAX*)input)->Y, ((SOFTMAX*)input)->step);
                
            case INS_SOFTMAX_LOG:
                return new SOFTMAX_LOG_GRADS(((SOFTMAX_LOG*)input)->X, ((SOFTMAX_LOG*)input)->Y, ((SOFTMAX_LOG*)input)->step);
                
            case INS_SOFTMAX_CE:
                return new SOFTMAX_CE_GRADS(((SOFTMAX_CE*)input)->X, ((SOFTMAX_CE*)input)->Y, ((SOFTMAX_CE*)input)->step);
    
            case INS_SWISH:
                return new SWISH_GRADS(((SWISH*)input)->X, ((SWISH*)input)->Y, ((SWISH*)input)->beta, 1, 1);
                
            case INS_GRADS_CLIPPED_RELU:
                return new CLIPPED_RELU_GRADS(((CLIPPED_RELU*)input)->X, ((CLIPPED_RELU*)input)->Y, ((CLIPPED_RELU*)input)->threshold, 1, 1);
                
            case INS_FLATTEN:
                return new FLATTEN_GRADS(((FLATTEN*)input)->X, ((FLATTEN*)input)->Y);
        
            default:
                logFatal(io::LOG_SEG_COMP,"No backward mapping for : " + std::to_string(input->opCode));
                input->print();
                throw std::runtime_error("No backward mapping for : " + std::to_string(input->opCode));
        }
    }
    
    void generateGrads(vector<Operation *> &forwardOps, vector<Operation *> &backwardOps) {
        //TODO: add autograd optimizations
        for (int i = (int)forwardOps.size() - 1; i >= 0; i--) {
            Operation* op = forwardOps[i];
            Operation* bwOp = getCorrespondBcwd(op);
            bwOp->bind(forwardOps[i]->params);
            backwardOps.push_back(bwOp);
        }
    }
}