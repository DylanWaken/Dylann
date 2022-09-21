//
// Created by Dylan on 9/19/2022.
//

#ifndef DYLANN_RESNET_CUH
#define DYLANN_RESNET_CUH

#include "../dylann/module/Module.cuh"

namespace dylann {
    struct ResnetIdentity : public Module {
    public:
        ResnetIdentity() = default;
        
        cuTensor forward(cuTensor& X) override {
            auto Y = conv2D(X, 3, 3, (int)X.desc().sizes.c, 1, 1, 1, 1, 1, 1);
            Y = batchnorm(Y, 1e-8, 0.1);
            Y = relu(Y);
            Y = conv2D(Y, 3, 3, (int)X.desc().sizes.c, 1, 1, 1, 1, 1, 1);
            Y = batchnorm(Y, 1e-8, 0.1);
            Y += X;
            return relu(Y);
        }
    };
    
    //the conv shortcut that allows changing dimensions.
    //the dimensions of features is cut down by 2 while the output channels is doubled
    struct ResnetConv : public Module {
    public:
        ResnetConv() = default;
        
        cuTensor forward(cuTensor& X) override {
            auto Y = conv2D(X, 3, 3, (int)X.desc().sizes.c * 2, 2, 2, 1, 1, 1, 1);
            Y = batchnorm(Y, 1e-8, 0.1);
            Y = relu(Y);
            Y = conv2D(Y, 3, 3, (int)X.desc().sizes.c * 2, 1, 1, 1, 1, 1, 1);
            Y = batchnorm(Y, 1e-8, 0.1);
            
            auto X_ = conv2D(X, 1, 1, (int)X.desc().sizes.c * 2, 2, 2, 0, 0, 1, 1);
            X_ = batchnorm(X_, 1e-8, 0.1);
            
            Y += X_;
            return relu(Y);
        }
    };
    
    struct ResnetBottleNeck : public Module {
    public:
        ResnetBottleNeck() = default;
        
        cuTensor forward(cuTensor& X) override {
            auto Y = conv2D(X, 1, 1, (int)X.desc().sizes.c / 4, 1, 1, 0, 0, 1, 1);\
            Y = batchnorm(Y, 1e-8, 0.1);
            Y = relu(Y);
            Y = conv2D(Y, 3, 3, (int)X.desc().sizes.c / 4, 1, 1, 1, 1, 1, 1);
            Y = batchnorm(Y, 1e-8, 0.1);
            Y = relu(Y);
            Y = conv2D(Y, 1, 1, (int)X.desc().sizes.c, 1, 1, 0, 0, 1, 1);
            Y = batchnorm(Y, 1e-8, 0.1);
            Y += X;
            return relu(Y);
        }
    };
    
    //the joint bottleneck that allows changing dimensions.
    //the dimensions of features is cut down by 2 while the output channels is doubled
    struct ResnetBottleNeckJoint : public Module {
    public:
        ResnetBottleNeckJoint() = default;
        
        cuTensor forward(cuTensor& X) override {
            auto Y = conv2D(X, 1, 1, (int)X.desc().sizes.c / 4, 0, 0, 1, 1, 1, 1);
            Y = batchnorm(Y, 1e-8, 0.1);
            Y = relu(Y);
            Y = conv2D(Y, 3, 3, (int)X.desc().sizes.c / 4, 2, 2, 1, 1, 1, 1);
            Y = batchnorm(Y, 1e-8, 0.1);
            Y = relu(Y);
            Y = conv2D(Y, 1, 1, (int)X.desc().sizes.c * 2, 0, 0, 1, 1, 1, 1);
            Y = batchnorm(Y, 1e-8, 0.1);
            
            auto X_ = conv2D(X, 1, 1, (int)X.desc().sizes.c * 2, 2, 2, 0, 0, 1, 1);
            X_ = batchnorm(X_, 1e-8, 0.1);
            
            Y += X_;
            return relu(Y);
        }
    };
} // dylann

#endif //DYLANN_RESNET_CUH
