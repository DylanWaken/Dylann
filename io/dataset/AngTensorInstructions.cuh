//
// Created by Dylan on 9/24/2022.
//

#ifndef DYLANN_ANGTENSORINSTRUCTIONS_CUH
#define DYLANN_ANGTENSORINSTRUCTIONS_CUH

#include "Data.cuh"

namespace io {
    struct AugmentInsTensor {
    public:
        virtual void process(Data& dataIn) = 0;
    };
    
    struct UniformNorm : public AugmentInsTensor {
        float min;
        float max;
        
        UniformNorm(float min, float max) {
            this->min = min;
            this->max = max;
        }
        
        void process(Data& dataIn) override;
    };
    
    struct StdNorm : public AugmentInsTensor {
        float mean;
        float std;
        
        StdNorm(float mean, float std) {
            this->mean = mean;
            this->std = std;
        }
        
        void process(Data& dataIn) override;
    };
    
} // io

#endif //DYLANN_ANGTENSORINSTRUCTIONS_CUH
