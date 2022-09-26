//
// Created by Dylan on 9/23/2022.
//

#ifndef DYLANN_AUGCVINSTRUCTIONS_CUH
#define DYLANN_AUGCVINSTRUCTIONS_CUH

#include <opencv2/opencv.hpp>

namespace io {
    struct AugmentInsCV {
        virtual cv::Mat augment(cv::Mat& imgIn) = 0;
    };
}


#endif //DYLANN_AUGCVINSTRUCTIONS_CUH
