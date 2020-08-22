#ifndef DTRACKER_H
#define DTRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "../OD/yolo_v2_class.hpp"


struct StateType {
	float dx, dy;
};

#define TRACK_TYPE_OPT 1
#define TRACK_TYPE_MED 2
#define TRACK_TYPE_KAL 3

#define BAD_PREDICT		-1
#define PREDICT_SUCESS	0

class DTracker
{
public:
	DTracker();
	DTracker(std::vector<cv::Point2f> pts, cv::Rect obj, int classId);


	int m_time_since_update;
	int m_hits;
	int m_hit_streak;
	int m_age;
	int m_id;
	int m_classId;
	std::string m_string_id;
//	Custom_Info m_custom_info;

	cv::Rect2f m_latestPos;
	int m_type;

	static int id_seed;

	cv::Rect2f m_smoothLastePos;
	static float LPF_Beta;
	cv::Rect get_lastest_position();
	void update_lastest_position(cv::Rect _lastest);

	int predict(std::vector<cv::Point2f> old_good_pts, std::vector<cv::Point2f> predict_good_pts,
				cv::Mat &gme_forward_trans);
	void update(std::vector<cv::Point2f> newPts, cv::Rect trueObj, cv::Mat &gme_forward_trans);

private:
	void _init();
	void _predict();
	void _update();
	std::vector<cv::Point2f> m_old_points;
	int getTrackType(std::vector<cv::Point2f> &pts);
	void init_kf();
	cv::KalmanFilter kf;
	cv::Mat measurement;
};

#endif // DTRACKER_H
