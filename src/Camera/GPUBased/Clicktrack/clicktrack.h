#ifndef CLICKTRACK_H
#define CLICKTRACK_H

#include "platedetector.h"
#include "../OD/yolo_v2_class.hpp"
#include "preprocessing.h"
#include "recognition.h"

#define PLATE_SUCCESS 1
#define PLATE_RUNNING 2
#define PLATE_FAIL    3
#define PLATE_DETECTED 4
#define Detector YoloDetector
class ClickTrack
{
    PlateDetector *m_plateDetector;
public:
    ClickTrack();
    void setDetector(Detector *_detector);
    std::string getPlateNumber_I420(image_t input, cv::Point clickPoint, int trackSize);

    void resetSequence();
    void setOCR(OCR* _OCR);

    ////
    /// \brief getPlateNumber_I420
    /// \param input
    /// \param h_gray
    /// \param objectPosition
    /// \return plateNumber
    ///
    int updateNewImage_I420(image_t input, cv::Mat h_gray, cv::Rect objectPosition, cv::Mat bgr_img);
    std::string getPlateNumber_I420();

public:
        int status = PLATE_FAIL;
private:
    std::string result;
    OCR* m_recognizor;
};

#endif // CLICKTRACK_H
