//
// Created by Dylan on 9/23/2022.
//

#include "AugCVInstructions.cuh"

cv::Mat io::RandRotate::augment(cv::Mat &imgIn) {
    default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
    uniform_real_distribution<float> distribution(1, 1.5);
    uniform_real_distribution<float> distribution2(0, 360);
    
    float scale = distribution(generator);
    float angle = distribution2(generator);
    
    auto rotMat = cv::getRotationMatrix2D(cv::Point2f((float)imgIn.cols / 2, (float)imgIn.rows / 2), angle, scale);
    cv::Mat imgOut;
    cv::warpAffine(imgIn, imgOut, rotMat, imgIn.size());
    
    return imgOut;
}

//TODO: recover this function
cv::Mat io::RandFlip::augment(cv::Mat &imgIn) {
    default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> distribution(0, 1);
    uniform_int_distribution<int> distribution2(0, 3);
    
    int flipDR = distribution(generator);
    int flipDec = distribution2(generator);
    
    flipDR = 0;
    
    cv::Mat imgOut;
    if(flipDec > 1){
        cv::flip(imgIn, imgOut, flipDR);
    } else {
        imgOut = imgIn;
    }
    
    return imgOut;
}

cv::Mat io::RandScaleCorp::augment(cv::Mat &imgIn) {
    default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> distributionY(0, (int)(scaleRatio * imgIn.rows - imgIn.rows));
    uniform_int_distribution<int> distributionX(0, (int)(scaleRatio * imgIn.cols - imgIn.cols));
    
    cv::Mat imgOut;
    cv::resize(imgIn, imgOut, cv::Size(), scaleRatio, scaleRatio);
    
    int x = distributionX(generator);
    int y = distributionY(generator);
    
    cv::Rect rect(x, y, (int)(imgOut.cols / scaleRatio), (int)(imgOut.rows / scaleRatio));
    imgOut = imgOut(rect);
    
    return imgOut;
}

cv::Mat io::RandPadCorp::augment(cv::Mat &imgIn) {
    default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> distributionY(0, padPixels * 2 - 1);
    uniform_int_distribution<int> distributionX(0, padPixels * 2 - 1);
    
    int x = distributionX(generator);
    int y = distributionY(generator);
    
    cv::Mat imgProc, imgOut;
    cv::copyMakeBorder(imgIn, imgProc, padPixels, padPixels, padPixels, padPixels, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    
    cv::Rect rect(x, y, imgIn.cols, imgIn.rows);
    imgOut = imgProc(rect);
    
    return imgOut;
}
