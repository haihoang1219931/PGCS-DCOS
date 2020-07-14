#include "ipsearch_oppoColor.h"

namespace ip {
namespace objsearch {

OppoColor::OppoColor() {}

OppoColor::~OppoColor(){}


void OppoColor::compute(cv::Mat& _inputImg, std::vector<cv::KeyPoint>& _keypoints, cv::Mat& _descriptors, cv::Ptr<cv::ORB> _descriptorsExtractor)
{
    if (_inputImg.empty() || _keypoints.empty())
    {
        _descriptors.release();
        return;
    }

    computeIml(_inputImg, _keypoints, _descriptors, _descriptorsExtractor);

}


void OppoColor::convertBRG2OpponentColorSpace(cv::Mat& _inbrgImg, std::vector<cv::Mat>& _opponentChannels)
{

    CV_Assert(_inbrgImg.type() == CV_8UC3);

    // compute opponent color space matrix
    _opponentChannels.resize(3);
    _opponentChannels[0] = cv::Mat(_inbrgImg.size(), CV_8UC1);  // red-green
    _opponentChannels[1] = cv::Mat(_inbrgImg.size(), CV_8UC1);  // r + g -2b
    _opponentChannels[2] = cv::Mat(_inbrgImg.size(), CV_8UC1);  // r + g + b




    for (int y = 0; y < _inbrgImg.rows; y++)
    {
        for( int x = 0; x < _inbrgImg.cols; x++)
        {
            cv::Vec3b v = _inbrgImg.at<cv::Vec3b>(y, x);
            uchar& b = v[0];
            uchar& g = v[1];
            uchar& r = v[2];

            _opponentChannels[0].at<uchar>(y,x) = cv::saturate_cast<uchar>(0.5f * (255 + g - r));
            _opponentChannels[1].at<uchar>(y,x) = cv::saturate_cast<uchar>(0.25f * (510 + r + g - 2*b));
            _opponentChannels[2].at<uchar>(y,x) = cv::saturate_cast<uchar>(1.f/3.f * (r + g +b));
        }
    }

}

struct KP_LessThan
{
    KP_LessThan(std::vector<cv::KeyPoint>& _kp): kp(&_kp){}
    bool operator()(int i, int j) const
    {
        return (*kp)[i].class_id < (*kp)[j].class_id;
    }

    const std::vector<cv::KeyPoint>* kp;
};


void OppoColor::computeIml(cv::Mat & _inbrgImg, std::vector<cv::KeyPoint> & _keypoints, cv::Mat & _descriptors, cv::Ptr<cv::ORB> _descriptorsExtractor)
{

    std::cout << "length of vector input keypoints " << _keypoints.size() << std::endl;


    std::vector<cv::Mat> opponentChannels;
    convertBRG2OpponentColorSpace(_inbrgImg, opponentChannels);

    const int N = 3;
    std::vector<cv::KeyPoint> channelKeypoints[N];
    cv::Mat channelDescriptors[N];
    std::vector<int> idxs[N];

    // compute descriptors three times

    int maxKeypointsCount = 0;
    for (int ci = 0; ci < N; ci++ )
    {
        channelKeypoints[ci].insert(channelKeypoints[ci].begin(), _keypoints.begin(), _keypoints.end());
        // use class_id member to get indices
        for (size_t ki = 0; ki < channelKeypoints[ci].size(); ki++)
            channelKeypoints[ci][ki].class_id = (int)ki;

        _descriptorsExtractor->compute(opponentChannels[ci], channelKeypoints[ci], channelDescriptors[ci]);
        idxs[ci].resize(channelKeypoints[ci].size());

        for(size_t ki = 0; ki < channelKeypoints[ci].size(); ki++)
        {
            idxs[ci][ki] = (int)ki;
        }

        std::sort(idxs[ci].begin(), idxs[ci].end(), KP_LessThan(channelKeypoints[ci]));
        maxKeypointsCount = std::max(maxKeypointsCount, (int)channelKeypoints[ci].size());
    }

    std::vector<cv::KeyPoint> outKeypoints;
    outKeypoints.reserve(_keypoints.size());

    int dSize = _descriptorsExtractor->descriptorSize();
    cv::Mat mergedDescriptors(maxKeypointsCount, 3*dSize, _descriptorsExtractor->descriptorType());
    int mergedCount = 0;

    // cp - current channel position
    size_t cp[] = {0, 0, 0};
    while(cp[0] < channelKeypoints[0].size() &&
          cp[1] < channelKeypoints[1].size() &&
          cp[2] < channelKeypoints[2].size())
    {
        const int maxInitIdx = std::max(0, std::max(channelKeypoints[0][idxs[0][cp[0]]].class_id,
                std::max( channelKeypoints[1][idxs[1][cp[1]]].class_id,
                channelKeypoints[2][idxs[2][cp[2]]].class_id)));

        while(channelKeypoints[0][idxs[0][cp[0]]].class_id < maxInitIdx && cp[0] < channelKeypoints[0].size()) {cp[0]++;}
        while(channelKeypoints[1][idxs[1][cp[1]]].class_id < maxInitIdx && cp[1] < channelKeypoints[1].size()) {cp[1]++;}
        while(channelKeypoints[2][idxs[2][cp[2]]].class_id < maxInitIdx && cp[2] < channelKeypoints[2].size()) {cp[2]++;}

        if(cp[0] >= channelKeypoints[0].size() || cp[1] >= channelKeypoints[1].size() || cp[2] >= channelKeypoints[2].size())
            break;

        if( channelKeypoints[0][idxs[0][cp[0]]].class_id == maxInitIdx &&
                channelKeypoints[1][idxs[1][cp[1]]].class_id == maxInitIdx &&
                channelKeypoints[2][idxs[2][cp[2]]].class_id == maxInitIdx )
        {
            outKeypoints.push_back(_keypoints[maxInitIdx]);
            // merge descriptors
            for(int ci = 0; ci < N; ci++)
            {
                cv::Mat dst = mergedDescriptors(cv::Range(mergedCount, mergedCount + 1), cv::Range(ci*dSize, (ci+1)*dSize));
                channelDescriptors[ci].row(idxs[ci][cp[ci]]).copyTo(dst);
                cp[ci]++;
            }
            mergedCount++;
        }
    }

     std::cout << "length of vector output keypoints " << outKeypoints.size() << std::endl;

    mergedDescriptors.rowRange(0, mergedCount).copyTo(_descriptors);

    std::cout << "length of vector output keypoints " << _descriptors.rows << " x " << _descriptors.cols << std::endl;
    std::swap(outKeypoints, _keypoints);

    for (int i = 0; i < N; i++)
    {
        channelKeypoints[i].clear();
        channelKeypoints[i].shrink_to_fit();
        idxs[i].clear();
        idxs[i].shrink_to_fit();
    }
    opponentChannels.clear();
    opponentChannels.shrink_to_fit();
}
}

}
