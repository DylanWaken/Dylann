//
// Created by Dylan on 9/23/2022.
//

#ifndef DYLANN_AUGCVINSTRUCTIONS_CUH
#define DYLANN_AUGCVINSTRUCTIONS_CUH

#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
namespace io {
    struct AugmentInsCV {
        virtual cv::Mat augment(cv::Mat& imgIn) = 0;
    };
    
    struct RandRotate : public AugmentInsCV {
    public:
        cv::Mat augment(cv::Mat& imgIn) override;
    };
    
    struct RandFlip : public AugmentInsCV {
    public:
        cv::Mat augment(cv::Mat& imgIn) override;
    };
    
    struct RandScaleCorp : public AugmentInsCV {
    public:
        float scaleRatio;
        explicit RandScaleCorp(float ScaleRatio) {
            this->scaleRatio = ScaleRatio;
        }
        
        cv::Mat augment(cv::Mat& imgIn) override;
    };
    
    struct RandPadCorp : public AugmentInsCV {
    public:
        int padPixels;
        explicit RandPadCorp(int PadPixels) {
            this->padPixels = PadPixels;
        }
        
        cv::Mat augment(cv::Mat& imgIn) override;
    };
}


#endif //DYLANN_AUGCVINSTRUCTIONS_CUH
