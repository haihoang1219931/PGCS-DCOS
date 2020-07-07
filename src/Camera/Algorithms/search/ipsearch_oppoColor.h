#ifndef IPSEARCH_OPPOCOLOR_H
#define IPSEARCH_OPPOCOLOR_H

#include <opencv2/opencv.hpp>
#include "ipsearch_utils.h"

namespace ip {
namespace objsearch {

    class OppoColor
    {
    public:
        OppoColor();
        ~OppoColor();

    public:
        void compute(cv::Mat & inputImg, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, cv::Ptr<cv::ORB> descriptorsExtractor);

        void convertBRG2OpponentColorSpace(cv::Mat& inbrgImg, std::vector<cv::Mat>& opponentChannels);

        void computeIml(cv::Mat& inbrgImg, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, cv::Ptr<cv::ORB> descriptorsExtractor);

    };

}


}

#endif // IPSEARCH_OPPOCOLOR_H
