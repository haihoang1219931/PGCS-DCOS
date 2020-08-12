#ifndef THRESHOLDING_HPP
#define THRESHOLDING_HPP
#include <opencv2/opencv.hpp>
#include"kalman.hpp"
#include <chrono>
//#include "Utilities.hpp"
//#include "LME/lme.hpp"
#include "ITrack.hpp"
#define THRESH_VAL_MIN 190
#define THRESH_VAL_MAX 210
#define MAX_DIST_RATIO 7      // max ratio allow between the distance object move in 3 consecutive frames
#define MAX_RADIUS 80
#define max_variance 0.12

class ThresholdingTracker
{
    private:
        bool found_contour;   //   flag to check if tracker found contour in search window or not
        std::vector<cv::Point2f> prev_vec, cur_vec; // vector indicate motion of centroid of contour found in 2 consecutive frame
        cv::Point2f currShift, prevShift;
        float object_perimeter = 100000;
        std::vector<std::vector<cv::Point>> contour;
        std::vector<cv::Vec4i> hierarchy;
        std::vector<cv::Rect> contour_bbox;
        cv::Point2f object_center;
        kalmanfilter motion_detector;
        int trackLostCnt = 0;

    public:
        cv::Mat current_frame, gray_frame, thresh_frame;
        cv::Rect selectedRoi;
        cv::Rect m_objRoi;
        int m_trackStatus;
        bool m_trackInited;
        int m_trackLostCnt;
        bool m_running;
    public:
        ThresholdingTracker();
        ~ThresholdingTracker();
        void initTrack(cv::Mat &_image, cv::Rect _selRoi);
        void performTrack(cv::Mat &_image);
        cv::Rect getPosition();
        bool isInitialized();
        bool isRunning();
        void resetTrack();
        int trackStatus();
    public:
        cv::Mat contrastEnhance(cv::Mat image, float weightingParam);
    private:
        bool checkPatchDetail(cv::Mat &patch);

};

#endif // THRESHOLDING_HPP
