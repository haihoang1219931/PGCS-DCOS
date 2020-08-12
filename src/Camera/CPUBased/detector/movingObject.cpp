

#include "movingObject.hpp"

//!
//! \brief MovingDetector::MovingDetector
//! \param _winSize
//!
MovingDetector::MovingDetector(const cv::Size &_winSize)
{
    m_winSize   = _winSize;
    m_mask      = cv::Mat::zeros(m_winSize, CV_8UC1);
    m_kernel    = cv::getStructuringElement(CV_SHAPE_ELLIPSE, cv::Size(MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE));
    m_firstFrame = true;
    pMOG2 = cv::createBackgroundSubtractorMOG2(MOG_HISTORY, MOG_THRESHOLD, false);
}


//!
//! \brief MovingDetector::~MovingDetector
//!
MovingDetector::~MovingDetector()
{
    pMOG2.release();
    m_mask.release();
}


//!
//! \brief setMinObjArea
//! \param _minObjArea
//!
void MovingDetector::setMinObjArea(const int &_minObjArea)
{
    assert(_minObjArea > 0);
    m_minObjArea = _minObjArea;
}


//!
//! \brief MovingDetector::setWindowSize
//! \param _winSize
//!
void MovingDetector::setWindowSize(const cv::Size &_winSize)
{
    pMOG2.release();
    m_mask.release();
    m_winSize = _winSize;
    m_mask = cv::Mat::zeros(m_winSize, CV_8UC1);
    pMOG2 = cv::createBackgroundSubtractorMOG2(MOG_HISTORY, MOG_THRESHOLD, false);
    m_firstFrame = true;
}


//!
//! \brief MovingDetector::process
//! \param _frame
//! \param _objLoc
//! \return
//!
bool MovingDetector::process(const cv::Mat &_frame, cv::Rect &_objLoc)
{
    int     top = (_frame.size().height - m_winSize.height) / 2,
            left = (_frame.size().width - m_winSize.width) / 2;

    if (left < 0) {
        left = 0;
        m_winSize.width = _frame.cols;
    }

    if (top < 0) {
        top = 0;
        m_winSize.height = _frame.rows;
    }

    cv::Mat patch = _frame(cv::Rect(left, top, m_winSize.width, m_winSize.height)).clone();

    if (patch.channels() > 1) {
        cv::cvtColor(patch, patch, cv::COLOR_RGB2GRAY);
    }

    //===== Background subtraction
    pMOG2->apply(patch, m_mask);

    if (m_firstFrame) {
        m_firstFrame = false;
        _objLoc = cv::Rect(_frame.cols / 2, _frame.rows / 2, m_winSize.width, m_winSize.height);
        return false;
    }

    // Noise filtering
    cv::Mat filtMask = m_mask.clone();
    cv::erode(filtMask, filtMask, m_kernel, cv::Point(-1, -1), MORPH_ERODE_ITERS);
    cv::dilate(filtMask, filtMask, m_kernel, cv::Point(-1, -1), MORPH_DILATE_ITERS);
    //===== Hotspot object detection
    cv::Mat brightPatch;
    patch.copyTo(brightPatch, filtMask);
    cv::Mat binIm;
    cv::threshold(brightPatch, binIm, 220, 255, cv::ADAPTIVE_THRESH_MEAN_C);
    cv::dilate(binIm, binIm, m_kernel, cv::Point(-1, -1), 1);
    //===== Detect the biggest moving region
    cv::Mat labelim, stats, centroids;
    int label = cv::connectedComponentsWithStats(binIm, labelim, stats, centroids, 4);

    if (label == 1) {
        _objLoc = cv::Rect(_frame.cols / 2, _frame.rows / 2, m_winSize.width, m_winSize.height);
        return false;
    } else {
        int maxarea = stats.at<int>(1, 4);
        int index = 1;

        for (int i = 1; i < label ; i++) {
            int area = stats.at<int>(i, 4);

            if (area >= maxarea) {
                maxarea = area;
                index = i;
            }
        }

        if (stats.at<int>(index, 4) < m_minObjArea) {
            _objLoc = cv::Rect(_objLoc = cv::Rect(_frame.cols / 2, _frame.rows / 2, m_winSize.width, m_winSize.height));
            return false;
        }

        cv::Mat rect = stats.row(index);
        _objLoc = cv::Rect(rect.at<int>(0), rect.at<int>(1), rect.at<int>(2), rect.at<int>(3));
        _objLoc.x += left;
        _objLoc.y += top;
    }

    return true;
}


void MovingDetector::reset()
{
    pMOG2.release();
    m_mask.release();
    m_mask = cv::Mat::zeros(m_winSize, CV_8UC1);
    pMOG2 = cv::createBackgroundSubtractorMOG2(MOG_HISTORY, MOG_THRESHOLD, false);
    m_firstFrame = true;
}
