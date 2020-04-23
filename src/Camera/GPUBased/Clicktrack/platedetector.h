#ifndef PLATEDETECTOR_H
#define PLATEDETECTOR_H

#include "../OD/yolo_v2_class.hpp"

class PlateDetector
{
	Detector* roi_detector;
	cv::Rect roi_detect;
public:
    PlateDetector();
    ~PlateDetector();
    void setDetector(Detector *_detector);
    bool detect_RGBA(image_t input, cv::Point clickPoint, int trackSize, bbox_t &box_to_track);
    bool detect_I420(image_t input, cv::Point clickPoint, int trackSize, bbox_t &box_to_track);
    bool detect_I420(image_t input, cv::Rect trackBox, bbox_t &box_to_track);
private:
    bool is_inside(cv::Rect rect, cv::Size imgSize);
	bool eliminate_box(std::vector<bbox_t>& boxs, int trackSize);
    bool eliminate_box(std::vector<bbox_t>& boxs, cv::Rect trackBox);
	bool select_best_box_to_track(std::vector<bbox_t>& boxs, bbox_t& best_box, cv::Point clickPoint, const int trackSize, bool filter = false);
    bool select_best_box_to_track(std::vector<bbox_t>& boxs, bbox_t& best_box, cv::Rect trackBox, bool filter = false);
	void getRegion(cv::Point clickPoint, int trackSize, cv::Size frameSize);
    void getRegion(cv::Rect trackBox, cv::Size frameSize);

};

#endif // PLATEDETECTOR_H
