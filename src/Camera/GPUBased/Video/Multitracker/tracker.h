#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "yolo_v2_class.hpp"
#include "plate_utils.h"

//struct Custom_Info {
//	std::vector<std::vector<Char>> codeTable;

//	std::string stringinfo;
//	int age;
//	float prob;

//	Custom_Info () {
//		Char c;
//		c.c = 'W';
//		c.score = 0;
//		std::vector<Char> vc;
//		vc.push_back(c);
//		for(int i = 0; i < 9; i++)
//			codeTable.push_back(vc);

//		age = 0;
//	}
//};

struct StateType {
	float dx, dy;
};

#define TRACK_TYPE_OPT 1
#define TRACK_TYPE_MED 2
#define TRACK_TYPE_KAL 3

#define BAD_PREDICT		-1
#define PREDICT_SUCESS	0

class Tracker
{
public:
	Tracker();
	Tracker(std::vector<cv::Point2f> pts, cv::Rect obj, int classId);


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

	cv::Rect get_lastest_position();

	int predict(std::vector<cv::Point2f> old_good_pts, std::vector<cv::Point2f> predict_good_pts,
				std::vector<cv::Point2f> old_prej_pts, cv::Mat &gme_forward_trans);
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

#endif // TRACKER_H
