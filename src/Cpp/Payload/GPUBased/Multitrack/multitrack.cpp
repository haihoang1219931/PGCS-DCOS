#include "multitrack.h"
#include <chrono>



#ifdef DEBUG
cv::Scalar detectColor = cv::Scalar(255, 50 ,0);
cv::Scalar predictColor = cv::Scalar(50, 255, 0);

bool isDebug()
{
		std::string key = "DEBUG";
		char * val = std::getenv(key.c_str());
		if (val == NULL) return false;
		if (std::string(val) == "1") return true;
		return false;
}

#endif

// Computes IOU between two bounding boxes
double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}
// Computes IOU between two bounding boxes
double GetIOU(bbox_t b1, bbox_t b2)
{
	cv::Rect_<float> bb_test(b1.x, b1.y, b1.w, b1.h);
	cv::Rect_<float> bb_gt(b2.x, b2.y, b2.w, b2.h);
	auto in_img = bb_test & bb_gt;
	if (in_img == bb_test || in_img == bb_gt) return 1;

	float in = (in_img).area();
	float un = bb_test.area() + bb_gt.area() - in;



	if (un < DBL_EPSILON)
		return 0;
	return (double)(in / un);
}

#define     GOODFEATURE_QUALITY     0.005f
#define     GOODFEATURE_MIN_DIS     10.0f

void MultiTrack::init()
{
	DTracker::id_seed = 0;
#ifdef DEBUG
	// for debug
	cv::RNG rng;
	for(int i = 0; i < 100; i++)
	{
			int r = rng.uniform(0, 256);
			int g = rng.uniform(0, 256);
			int b = rng.uniform(0, 256);
			colors.push_back(cv::Scalar(r,g,b));
	}
#endif

	gooddetector = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, 0, GOODFEATURE_QUALITY, GOODFEATURE_MIN_DIS);
#ifdef USE_FAST
	fastdetector = cv::cuda::FastFeatureDetector::create(20);
#endif
	d_pyrLK_spare = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(15, 15), 2, 10);
}

MultiTrack::MultiTrack()
{
	init();
}

void MultiTrack::renew()
{
	DTracker::id_seed = 0;
	trackers.clear();
}

MultiTrack::~MultiTrack() { }


bool isInside(cv::Point p, bbox_t&rec)
{
	if (p.x < rec.x) return false;
	if (p.x > rec.x + rec.w) return false;
	if (p.y < rec.y) return false;
	if (p.y > rec.y + rec.h) return false;
	return true;
}

#ifdef OPT_GPU

void download(const cv::cuda::GpuMat &d_mat, std::vector<cv::Point2f> & vec)
{
	vec.resize(d_mat.cols);
	cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

void download(const cv::cuda::GpuMat &d_mat, std::vector<uchar> & vec)
{
	vec.resize(d_mat.cols);
	cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

void upload(const std::vector<cv::Point2f> & vec,  cv::cuda::GpuMat &d_mat)
{
	cv::Mat mat(1, vec.size(), CV_32FC2, (void*)&vec[0]);
	d_mat.upload(mat);
}

#endif


std::vector<bbox_t> MultiTrack::run(cv::cuda::GpuMat& d_cur_gray, std::vector<bbox_t> detection_result,
									  cv::Mat gme_forward_trans,  cv::Mat draw_frame, const cv::Mat& cur_gray )
{
	bool drawable = !draw_frame.empty();

	auto result_vec = detection_result;
//	if (result_vec.empty()) return result_vec;
	// new: make detection a set of different object (same object have IoU > 85%)
	std::vector<bbox_t> result_set;
	for(auto v: result_vec)
	{
		bool continueFlag = false;
		for(auto s: result_set)
		{
			// check adjecent
			if (GetIOU(v, s) > 0.7)
			{
				continueFlag = true;
				break;
			}
		}
		if (continueFlag) continue;
		result_set.push_back(v);
	}
#ifdef DEBUG
//	std::cout << "result_vec: " << result_vec.size() << std::endl;
//	std::cout << "result_set: " << result_set.size() << std::endl;
#endif
	result_vec = result_set;
//	return result_vec;

#ifdef DEBUG
	if (drawable && isDebug())
	{
		cv::putText(draw_frame, "detection: " + std::to_string(result_vec.size()), cv::Point2f(10, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, detectColor, 2);
		for(auto b : result_vec)
			cv::rectangle(draw_frame, cv::Rect(b.x, b.y, b.w, b.h), detectColor, 2);
	}
#endif

	/////////////////////////////////////////////////////////
	/// \brief extract keypoints
	///
//	cv::Mat mask = cv::Mat::zeros(input.h, input.w, CV_8UC1);
	cv::Mat mask = cv::Mat::zeros(d_cur_gray.size(), CV_8UC1);

	for(auto i : result_vec)
//		if (std::find(ids.begin(), ids.end(), i.obj_id) != ids.end())
			cv::rectangle(mask, cv::Rect(i.x, i.y, i.w, i.h), cv::Scalar(255), -1);


	cv::cuda::GpuMat d_mask(mask);

	assert(d_cur_gray.size() == d_mask.size());

#ifndef USE_FAST
	gooddetector->detect( d_cur_gray, d_cur_pts, d_mask );
//	std::cout << "d_cur_pts: " << d_cur_pts.cols;
	download(d_cur_pts, cur_pts);
//	cv::cuda::GpuMat out_des;
//	fastdetector->detectAndCompute(d_cur_gray, d_mask, cur_pts, out_des);
//	std::cout << "      d_cur_pts: " << cur_pts.size();
//	cv::Mat debug_cur_gray(d_cur_gray);
//	extractKeypoints(debug_cur_gray, cur_pts, mask, 0);
//	std::cout << "      cur_pts: " << cur_pts.size() << std::endl;
//	std::cout << std::endl;
#else // USE_FAST
	std::vector<cv::KeyPoint> kp;
	fastdetector->detect(d_cur_gray, kp, d_mask);
	cv::KeyPoint::convert(kp, cur_pts);

	upload(cur_pts, d_cur_pts);
#endif


#ifdef DEBUG
	if (drawable && isDebug())
	{
		for(uint i = 0; i < cur_pts.size(); i++)
			cv::circle(draw_frame, cur_pts[i], 5, colors[i % 100], -1);
	}
#endif

	std::vector<int> cur_pts_id(cur_pts.size(), -1);
	std::vector<bool> isFlaged(cur_pts.size(), false);
	if (trackers.size() == 0)
	{
		for(auto b: result_vec)
		{
			std::vector<cv::Point2f> pts;
			std::vector<int> ids;
			for(uint i = 0; i < cur_pts.size(); i++)
				if (true != isFlaged[i] && isInside(cur_pts[i], b))
				{
					pts.push_back(cur_pts[i]);
					ids.push_back(i);
					isFlaged[i] = true;
				}

			DTracker tracker(pts, cv::Rect(b.x, b.y, b.w, b.h), b.obj_id);
			trackers.push_back(tracker);
			for(auto i: ids)
				cur_pts_id[i] = tracker.m_id;
		}
		old_gray = cur_gray.clone();
#ifdef OPT_GPU
		d_old_gray = d_cur_gray.clone();
		d_old_pts = d_cur_pts.clone();
#endif
		old_pts = cur_pts;
		old_pts_id = cur_pts_id;

//		return result_vec;

		frameTrackingResult.clear();
		for(auto it = trackers.begin(); it != trackers.end();)
		{
			bbox_t res;
			cv::Rect_<float> box = (*it).get_lastest_position();
			res.x = box.x;
			res.y = box.y;
			res.w = box.width;
			res.h = box.height;
			res.track_id = (*it).m_id + 1;
			res.obj_id = (*it).m_classId;

			res.track_info.stringinfo = (*it).m_string_id;
			frameTrackingResult.push_back(res);
			it++;
		}
		return frameTrackingResult;
	}

	/////////////////////////////////////////////////////////
    /// \brief forward optical flow.
	///

	// optical flow here
	std::vector<uchar> status;
	std::vector<float> err;
	cv::TermCriteria criteria =cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
	std::vector<cv::Point2f> predict_points, old_good_points;
	std::vector<int> predict_good_idx, old_good_idx;
	predict_good_idx.resize(old_pts.size(), -1);
	old_good_idx.resize(old_pts.size(), -1);

	cv::cuda::GpuMat d_predict_points;
	cv::cuda::GpuMat d_status;

//	cv::Mat old_cpu_gray, cur_cpu_gray;
//	d_old_gray.download(old_cpu_gray);
//	d_cur_gray.download(cur_cpu_gray);
//	cv::imshow("old_cpu_gray", cur_gray);
//	cv::imshow("cur_cpu_gray", cur_cpu_gray);
//	cv::waitKey();
	if (!old_pts.empty())
		d_pyrLK_spare->calc(d_old_gray, d_cur_gray, d_old_pts, d_predict_points, d_status);

	download(d_old_pts, old_pts);
	download(d_predict_points, predict_points);
	download(d_status, status);

	for(uint i = 0; i < old_pts.size(); i++)
	{
		if (status[i])
		{
			old_good_idx[i] = old_pts_id[i];
			predict_good_idx[i] = old_pts_id[i];
		}
	}
	// draw
//	std::cout << "....................... forward good point: " << predict_good_points.size() << std::endl;

	/////////////////////////////////////////////////////////
	/// \brief get predicted locations from existing trackers.
	///
	predictedBoxes.clear();
	std::vector<DTracker> predictedBoxesType;
	for(auto it = trackers.begin(); it != trackers.end();)
	{
		std::vector<cv::Point2f> old_good_pts, predict_good_pts;
		for(uint i = 0; i < old_pts.size(); i++)
		{
			if (old_good_idx[i] == it->m_id && predict_good_idx[i] == it->m_id)
			{
				old_good_pts.push_back( old_pts[i] );
				predict_good_pts.push_back( predict_points[i] );
			}
		}
//		std::cout << "------------------------------ predict_good_pts : " << predict_good_pts.size() << std::endl;
		if (PREDICT_SUCESS != it->predict(old_good_pts, predict_good_pts, gme_forward_trans))
			it = trackers.erase(it);
		else {
			predictedBoxes.push_back(it->get_lastest_position());
			predictedBoxesType.push_back(*it);
			it++;
		}
	}

#ifdef DEBUG
	if (drawable && isDebug())
	{
		cv::putText(draw_frame, "predictedBoxes: " + std::to_string(predictedBoxes.size()), cv::Point2f(10, 80), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
		std::vector<std::string> types {"OPT", "MED", "KAL"};
		for(int i = 0; i < predictedBoxes.size(); i++)
		{
			auto b = predictedBoxes[i];
			cv::rectangle(draw_frame, b, predictColor, 2);
			cv::putText(draw_frame, types[predictedBoxesType[i].m_type - 1],
						cv::Point(b.x, b.y - 3), CV_FONT_HERSHEY_COMPLEX, 1,
						predictColor, 2);
		}
	}
#endif

	/////////////////////////////////////////////////////////
	/// \brief associate detections to tracked object (both represented as bounding boxes)
	///

	trkNum = predictedBoxes.size();
	detNum = result_vec.size();
	iouMatrix.clear();
	iouMatrix.resize(trkNum, vector<double>(detNum, 0));
	for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
	{
		for (unsigned int j = 0; j < detNum; j++)
		{
			// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
			cv::Rect_<float> box(result_vec[j].x,
								 result_vec[j].y,
								 result_vec[j].w,
								 result_vec[j].h);
			iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], box);
		}
	}
	if (iouMatrix.empty()) return result_vec;
	// solve the assignment problem using hungarian algorithm.
	// the resulting assignment is [track(prediction) : detection], with len=preNum
	HungarianAlgorithm HungAlgo;
	assignment.clear();
	HungAlgo.Solve(iouMatrix, assignment);
	// find matches, unmatched_detections and unmatched_predictions
	unmatchedTrajectories.clear();
	unmatchedDetections.clear();
	allItems.clear();
	matchedItems.clear();
	if (detNum > trkNum) // there are unmatched detections
	{
		for (unsigned int n = 0; n < detNum; n++)
			allItems.insert(n);
		for (unsigned int i = 0; i < trkNum; ++i)
			matchedItems.insert(assignment[i]);
		set_difference(allItems.begin(), allItems.end(),
					   matchedItems.begin(), matchedItems.end(),
					   insert_iterator<set<uint>>(unmatchedDetections, unmatchedDetections.begin()));
	}
	else if (detNum < trkNum) // there are unmateched trajectory/predictions
	{
		for (unsigned int i = 0; i < trkNum; i++)
			if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
				unmatchedTrajectories.insert(i);
	}
	else;

	// filter out matched with low IOU
	matchedPairs.clear();
	for (unsigned int i = 0; i < trkNum; ++i)
	{
		if (assignment[i] == -1) // pass over invalid values
			continue;
		if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
		{
			unmatchedTrajectories.insert(i);
			unmatchedDetections.insert(assignment[i]);
		}
		else
			matchedPairs.push_back(cv::Point(i, assignment[i]));
	}

	/////////////////////////////////////////////////////////
	/// \brief updating trackers
	/// \details update matched trackers with assigned detections.
	/// each prediction is corresponding to a tracker
	///

	unsigned long detIdx, trkIdx;
	for(auto i : sort_indexes(matchedPairs, result_vec))
	{
		trkIdx = matchedPairs[i].x;
		detIdx = matchedPairs[i].y;
		std::vector<cv::Point2f> pts_to_update;
		for(uint j = 0; j < cur_pts.size(); j++)
		{
			if (true != isFlaged[j] && isInside(cur_pts[j], result_vec[detIdx]))
			{
				pts_to_update.push_back(cur_pts[j]);
				isFlaged[j] = true;
				// update cur_pts_id
				cur_pts_id[j] = trackers[trkIdx].m_id;
			}
		}
		cv::Rect_<float> box(result_vec[detIdx].x,
							 result_vec[detIdx].y,
							 result_vec[detIdx].w,
							 result_vec[detIdx].h);

		trackers[trkIdx].update(pts_to_update, box, gme_forward_trans);
	}

	/////////////////////////////////////////////////////////
	/// \brief create and initialise new trackers for unmatched detections
	///


	for (auto umd : unmatchedDetections)
	{
		std::vector<cv::Point2f> pts;
		std::vector<int> ids;
		bbox_t b = result_vec[umd];
		for(uint i = 0; i < cur_pts.size(); i++)
			if (true != isFlaged[i] && isInside(cur_pts[i], b))
			{
				pts.push_back(cur_pts[i]);
				ids.push_back(i);
				isFlaged[i] = true;
			}
		DTracker tracker(pts, cv::Rect(b.x, b.y, b.w, b.h), b.obj_id);
		trackers.push_back(tracker);
		for(auto i: ids)
			cur_pts_id[i] = tracker.m_id;
	}

	/////////////////////////////////////////////////////////
	/// \brief get trackers' output
	///

	frameTrackingResult.clear();
	for(auto it = trackers.begin(); it != trackers.end();)
	{
        if (((*it).m_time_since_update < 4) &&
                ((*it).m_hit_streak >= min_hits))
//        if (((*it).m_hit_streak >= min_hits))
		{
			bbox_t res;
			cv::Rect_<float> box = (*it).get_lastest_position();
                // check to make sure res is inside d_cur_gray
            if (box.x < 0 || box.y < 0 ||
                    box.x + box.width >= static_cast<unsigned int> (d_cur_gray.cols) ||
                    box.y + box.height >= static_cast<unsigned int> (d_cur_gray.rows))
            {
                it = trackers.erase(it);
            } else
            {
                res.x = static_cast<unsigned int>( box.x );
                res.y = static_cast<unsigned int>( box.y );
                res.w = static_cast<unsigned int>( box.width  );
                res.h = static_cast<unsigned int>( box.height );
                res.track_id = (*it).m_id + 1;
                res.obj_id = (*it).m_classId;

                res.track_info.stringinfo = (*it).m_string_id;
                frameTrackingResult.push_back(res);
                it++;
            }
		}
		else if ((*it).m_time_since_update > max_age)
			it = trackers.erase(it);
		else
			it++;
	}

	// recursive
	old_gray = cur_gray.clone();
#ifdef OPT_GPU
	d_old_gray = d_cur_gray.clone();
	d_old_pts = d_cur_pts;
#endif
	old_pts = cur_pts;
	old_pts_id = cur_pts_id;

	return frameTrackingResult;
}

