#ifndef LME_HPP
#define LME_HPP


#include <opencv2/opencv.hpp>


#define     GOOD_FEATURE            1 << 0
#define     FAST_CORNER             1 << 1

#define     HOMOGRAPHY              1 << 0
#define     RIGID_TRANSFORM         1 << 1

#define     GENERIC_ERROR           -1000
#define     SUCCESS                 0
#define     BAD_TRANSFORM           -1

#define     MAX_CORNERS             500u
#define     MIN_CORNERS             100u
#define     VERTICA_BLKS            3u
#define     HORIZON_BLKS            4u

#define     GOODFEATURE_QUALITY     0.005f
#define     GOODFEATURE_MIN_DIS     20.0f
#define     GOODFEATURE_BLKSIZE     3
#define     FASTCORNERS_MIN_DIS     35
#define     RANSAC_INLIER_THRESHOLD 2.0f
#define     MIN_INLIER              15
#define     MIN_INLIER_RATIO        0.1f
#define     MIN_EIGEN_VALUE         1e-4f


class LMEstimator{

    cv::Mat m_prevGray;
    cv::Mat m_currGray;
    std::vector<cv::Point2f> m_prevPts;
    std::vector<cv::Point2f> m_currPts;
    cv::Mat m_trans;

public:
    LMEstimator( ) { }
    ~LMEstimator( ) { }

private:
    void ageDelay();

public:
    int run( const cv::Mat &img, const cv::Rect &roi, float *dx, float *dy );
    cv::Mat getTrans();

};

#endif // LME_HPP

