#ifndef MOVINGOBJECT_HPP
#define MOVINGOBJECT_HPP

#include <opencv2/opencv.hpp>

#define MOG_HISTORY         300
#define MOG_THRESHOLD       20
#define MORPH_KERNEL_SIZE   5
#define MORPH_ERODE_ITERS   2
#define MORPH_DILATE_ITERS  3

class MovingDetector
{
    cv::Size m_winSize;
    int      m_minObjArea;
    cv::Mat  m_mask;
    cv::Mat  m_kernel;
    bool     m_firstFrame;
    cv::Ptr<cv::BackgroundSubtractorMOG2> pMOG2;

public:
    MovingDetector( const cv::Size &_winSize );
    ~MovingDetector();

    void setMinObjArea( const int &_minObjArea );
    void setWindowSize( const cv::Size &_winSize );
    bool process( const cv::Mat &_frame, cv::Rect &_objLoc );
    void reset();
};

#endif // MOVINGOBJECT_HPP
