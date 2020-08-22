#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <vector>
#include <opencv2/opencv.hpp>

enum gState{
    WAITING,
    COUNTING,
    UP,
    DOWN
};

//===============================
// I. Main Functions            "
//===============================
std::vector<cv::Mat> preprocess(cv::Mat grayImg, int plateType, int *sign);
// Input: grayscale image ==> Output: deskewed image
cv::Mat deskewImage(cv::Mat image);
// Input: grayscale image ==> Output: binary image
void    binarize(const cv::Mat input, cv::Mat &output, const int windowSize);
void    filter(const cv::Mat input, cv::Mat &output);
// Input: binary image ==> Output: a vector of characters
std::vector<cv::Mat> extractCharsLP(cv::Mat input, int *sign);
std::vector<cv::Mat> extractCharsSP(cv::Mat input, int *sign);
std::vector<cv::Mat> extractChars(cv::Mat input, int *sign);

//-----------------------------/-
//===============================
// II. Support Functions        "
//===============================
void    contrastEnhance(cv::Mat &src, cv::Mat &dst, int dist=10);
void    insertPadding(const cv::Mat input, cv::Mat &output, const int paddingSize, const int paddingValue);
void    filterBlobNoise(const cv::Mat input, cv::Mat &output, const int windowSize);
void    horiPrune(const cv::Mat input, cv::Mat &output, float thres);
void    vertPrune(const cv::Mat input, cv::Mat &output, int plateType);
//-----------------------------/-

#endif // PREPROCESSING_H
