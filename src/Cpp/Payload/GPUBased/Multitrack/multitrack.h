#ifndef MULTITRACK_H
#define MULTITRACK_H

#define OPT_GPU 1

#include "Hungarian.h"
//#include "gme_short.hpp"
#include "../OD/yolo_v2_class.hpp"
#include <numeric>

#include "Dtracker.h"

class MultiTrack
{
public:
    int min_hits = 0;
	int max_age = 10;
	double iouThreshold = 0.3;
	vector<cv::Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	set<uint> unmatchedDetections;
	set<uint> unmatchedTrajectories;
	set<uint> allItems;
	set<uint> matchedItems;
	vector<int> assignment;
	vector<cv::Point> matchedPairs;
	vector<bbox_t> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	// for track_optflow
	cv::Mat old_gray;
	std::vector<cv::Point2f> cur_pts, old_pts;
	std::vector<int> old_pts_id;

	std::vector<DTracker> trackers;

	cv::cuda::GpuMat d_old_gray;
	cv::Ptr<cv::cuda::CornersDetector> gooddetector;
#ifdef USE_FAST
	cv::Ptr<cv::cuda::FastFeatureDetector> fastdetector;
#endif
	cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_spare;
	cv::cuda::GpuMat d_old_pts, d_cur_pts;

public:
	MultiTrack();
	~MultiTrack();
	void renew();
    std::vector<bbox_t> run(cv::cuda::GpuMat& cur_gpu_gray, std::vector<bbox_t> detection_result,
										  cv::Mat gme_forward_trans = cv::Mat::eye(3, 3, CV_64F),  cv::Mat draw_frame = cv::Mat(), const cv::Mat& cur_gray = cv::Mat() );
private:
	std::vector<int> ids {4, 5, 6, 8, 9, 10};
	std::vector<uint> sort_indexes(const std::vector<cv::Point> &v, std::vector<bbox_t> &result_vec) {
		std::vector<int> ids = this->ids;
		std::vector<uint> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);
		std::sort(idx.begin(), idx.end(),
				  [&v, &result_vec, &ids](size_t i1, size_t i2) {
					if (std::find(ids.begin(), ids.end(), result_vec[v[i1].y].obj_id) == ids.end())
						return false;
					if (std::find(ids.begin(), ids.end(), result_vec[v[i2].y].obj_id) == ids.end())
						return true;
					int w1 = result_vec[v[i1].y].w * result_vec[v[i1].y].h;
					int w2 = result_vec[v[i2].y].w * result_vec[v[i2].y].h;
					return  w1 > w2;});
		return idx;
	}
	void init();

#ifdef DEBUG
private:
	vector<cv::Scalar> colors;
#endif
};

#endif // MULTITRACK_H
