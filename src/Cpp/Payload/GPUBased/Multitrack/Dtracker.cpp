#include "Dtracker.h"

#define     GENERIC_ERROR           -1000
#define     SUCCESS                 0
#define     BAD_TRANSFORM           -1

template<typename T>
T getMedianAndDoPartition(std::vector<T>& values)
{
	size_t size = values.size();
	if(size%2==0)
	{
		std::nth_element(values.begin(), values.begin() + size/2-1, values.end());
		T firstMedian = values[size/2-1];

		std::nth_element(values.begin(), values.begin() + size/2, values.end());
		T secondMedian = values[size/2];

		return (firstMedian + secondMedian) / (T)2;
	}
	else
	{
		size_t medianIndex = (size - 1) / 2;
		std::nth_element(values.begin(), values.begin() + medianIndex, values.end());

		return values[medianIndex];
	}
}
template<typename T>
T getMedian(const std::vector<T>& values)
{
	std::vector<T> copy(values);
	return getMedianAndDoPartition(copy);
}

cv::Rect2d vote(const std::vector<cv::Point2f>& oldPoints,const std::vector<cv::Point2f>& newPoints,const cv::Rect2d& oldRect, cv::Point2f& mD){
	cv::Rect2d newRect;
	cv::Point2d newCenter(oldRect.x+oldRect.width/2.0,oldRect.y+oldRect.height/2.0);
	const size_t n=oldPoints.size();

	if (n==1) {
		newRect.x=oldRect.x+newPoints[0].x-oldPoints[0].x;
		newRect.y=oldRect.y+newPoints[0].y-oldPoints[0].y;
		newRect.width=oldRect.width;
		newRect.height=oldRect.height;
		mD.x = newPoints[0].x-oldPoints[0].x;
		mD.y = newPoints[0].y-oldPoints[0].y;
		return newRect;
	}

	float xshift=0,yshift=0;
	std::vector<float> buf_for_location(n, 0.);
	for(size_t i=0;i<n;i++){  buf_for_location[i]=newPoints[i].x-oldPoints[i].x;  }
	xshift = getMedianAndDoPartition(buf_for_location);
	newCenter.x+=xshift;
	for(size_t i=0;i<n;i++){  buf_for_location[i]=newPoints[i].y-oldPoints[i].y;  }
	yshift = getMedianAndDoPartition(buf_for_location);
	newCenter.y+=yshift;
	mD = cv::Point2f((float)xshift,(float)yshift);

	std::vector<double> buf_for_scale(n*(n-1)/2, 0.0);
	for(size_t i=0,ctr=0;i<n;i++){
		for(size_t j=0;j<i;j++){
			double nd=norm(newPoints[i] - newPoints[j]);
			double od=norm(oldPoints[i] - oldPoints[j]);
			buf_for_scale[ctr]=(od==0.0)?0.0:(nd/od);
			ctr++;
		}
	}

	double scale = getMedianAndDoPartition(buf_for_scale);
	//	dprintf(("xshift, yshift, scale = %f %f %f\n",xshift,yshift,scale));
	newRect.x=newCenter.x-scale*oldRect.width/2.0;
	newRect.y=newCenter.y-scale*oldRect.height/2.0;
	newRect.width=scale*oldRect.width;
	newRect.height=scale*oldRect.height;
	//	dprintf(("rect old [%f %f %f %f]\n",oldRect.x,oldRect.y,oldRect.width,oldRect.height));
	//	dprintf(("rect [%f %f %f %f]\n",newRect.x,newRect.y,newRect.width,newRect.height));

	return newRect;
}

int estimateRigidTransform(std::vector<cv::Point2f> &pts_1, std::vector<cv::Point2f> &pts_2, cv::Mat &trans)
{
	if (pts_1.size() != pts_2.size()) return -1;
	auto convert = [&](cv::Mat &m)
	{
		cv::Mat result = cv::Mat::eye(3, 3, CV_64F);
		result.at<double>(0, 0) = m.at<float>(0, 0);
		result.at<double>(0, 1) = m.at<float>(0, 1);
		result.at<double>(0, 2) = m.at<float>(0, 2);
		result.at<double>(1, 0) = m.at<float>(1, 0);
		result.at<double>(1, 1) = m.at<float>(1, 1);
		result.at<double>(1, 2) = m.at<float>(1, 2);
		return result;
	};
	cv::Mat tmp_trans;
	cv::videostab::RansacParams ransacParams = cv::videostab::RansacParams::default2dMotion(cv::videostab::MM_SIMILARITY);
	ransacParams.thresh = 2.0f;
	ransacParams.eps = 0.5f;
	ransacParams.prob = 0.99f;
	ransacParams.size = 4;

	int num;
	try {
		tmp_trans = cv::videostab::estimateGlobalMotionRansac(pts_1, pts_2,
															  cv::videostab::MM_SIMILARITY, ransacParams,
															  nullptr, &num);
	}
	catch (const std::exception &e)
	{
		std::cerr << "* EXCEPTION: vtx_estimate_transform(): " << e.what() << std::endl;
		return GENERIC_ERROR;
	}
	if( !tmp_trans.empty() )
	{
		convert( tmp_trans ).copyTo( trans );
	}
	else
	{
		return BAD_TRANSFORM;
	}
	return SUCCESS;
}

cv::Point2f my_mul(cv::Mat trans, cv::Point2f in)
{
	// trans is CV_64F

	double x_ = trans.at<double>(0,0)*in.x + trans.at<double>(0,1)*in.y + trans.at<double>(0,2);
	double y_ = trans.at<double>(1,0)*in.x + trans.at<double>(1,1)*in.y + trans.at<double>(1,2);
	return cv::Point2f(x_, y_);
}

cv::Rect_<float> LMEBox(cv::Mat trans, cv::Rect_<float> inbox)
{
	std::vector<cv::Point2f> newPoligon;
	newPoligon.push_back(my_mul(trans, inbox.tl()));
	newPoligon.push_back(my_mul(trans, cv::Point(inbox.x, inbox.y + inbox.height)));
	newPoligon.push_back(my_mul(trans, inbox.br()));
	newPoligon.push_back(my_mul(trans, cv::Point(inbox.x + inbox.width, inbox.y)));
	cv::Rect rect = cv::boundingRect(newPoligon);
	return rect;
}

int DTracker::id_seed = 0;

void DTracker::init_kf()
{
	int stateNum = 4;
//	int stateNum = 2;
	int measureNum = 2;
	kf = cv::KalmanFilter(stateNum, measureNum, 0);

	measurement = cv::Mat::zeros(measureNum, 1, CV_32F);

//	kf.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<
//							1, 0, 0, 0,
//							0, 1, 0, 0,
//							0, 0, 1, 0,
//							0, 0, 0, 1);

	kf.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<
							1, 0, 0, 0,
							0, 1, 0, 0,
							0, 0, 1, 0,
							0, 0, 0, 1);

	setIdentity(kf.measurementMatrix);
	setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
	setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
	setIdentity(kf.errorCovPost, cv::Scalar::all(1));

	kf.statePost.setTo( cv::Scalar::all( 0.0 ) );
}

DTracker::DTracker()
{

}

float DTracker::LPF_Beta = 0.25f;

DTracker::DTracker(std::vector<cv::Point2f> pts, cv::Rect obj, int classId)
{
	_init();
	init_kf();
//	LPF_Beta = 0.25f;
	m_type = getTrackType(pts);
	m_old_points = pts;
	m_latestPos = obj;
	m_smoothLastePos = obj;
	m_classId = classId;
}

int DTracker::predict(std::vector<cv::Point2f> old_good_pts, std::vector<cv::Point2f> predict_good_pts, cv::Mat &gme_forward_trans)
{
//	std::cout << "predict_good_pts: " << predict_good_pts.size() << std::endl;
	_predict();
	m_type = getTrackType(predict_good_pts);
	switch (getTrackType(predict_good_pts)) {
	case TRACK_TYPE_OPT:
	{
		cv::Mat trans;
		if (SUCCESS != estimateRigidTransform(old_good_pts, predict_good_pts, trans) )
		{
			std::cout << "BAD_PREDICT" << std::endl;
//			return BAD_PREDICT;
			// kalman here
			{
				cv::Mat p = kf.predict();
				StateType predictionMovement;
				predictionMovement.dx = p.at<float>(0, 0);
				predictionMovement.dy = p.at<float>(1, 0);

				m_latestPos.x += predictionMovement.dx;
				m_latestPos.y += predictionMovement.dy;
			}
		}
		cv::Rect2f lmedPos;
		lmedPos = LMEBox(gme_forward_trans, m_latestPos);

		StateType measureMov;
		measureMov.dx = lmedPos.x + lmedPos.width /2.0 - m_latestPos.x - m_latestPos.width /2.0;
		measureMov.dy = lmedPos.y + lmedPos.height/2.0 - m_latestPos.y - m_latestPos.height/2.0;

		measurement.at<float>(0, 0) = measureMov.dx;
		measurement.at<float>(1, 0) = measureMov.dy;
		kf.correct(measurement);

		m_latestPos = lmedPos;
		update_lastest_position(m_latestPos);
		m_old_points = predict_good_pts;

		return PREDICT_SUCESS;
		break;
	}
//	case TRACK_TYPE_MED:
//	{
//		std::vector<float> FBerror;
//		for (size_t i = 0; i < predict_good_pts.size(); i++) {
//			float err = (float)cv::norm(old_good_pts[i] - old_prej_pts[i]);
//			FBerror.push_back(err);
//		}
//		float FBerrorMedian = getMedian(FBerror);
//		std::vector<bool> status=std::vector<bool>(old_good_pts.size(),true);
//		for(size_t i=0;i<old_good_pts.size();i++){
//			status[i] = status[i] && (FBerror[i] <= FBerrorMedian);
//		}

//		std::vector<cv::Point2f> pointsToTrackOld, pointsToTrackCur;
//		for(size_t i = 0; i < old_good_pts.size(); i++) {
//			if (status[i] == 1)
//			{
//				pointsToTrackOld.push_back(old_good_pts[i]);
//				pointsToTrackCur.push_back(predict_good_pts[i]);
//			}
//		}

//		std::vector<cv::Point2f> di(pointsToTrackOld.size());
//		for(size_t i=0; i<pointsToTrackOld.size(); i++){
//			di[i] = pointsToTrackCur[i]-pointsToTrackOld[i];
//		}
//		cv::Point2f mDisplacement;
//		cv::Rect newPos;
//		newPos = vote(pointsToTrackOld, pointsToTrackCur, m_latestPos, mDisplacement);

//		std::vector<float> displacements;
//		for(size_t i=0;i<di.size();i++){
//			di[i]-=mDisplacement;
//			displacements.push_back((float)sqrt(di[i].ddot(di[i])));
//		}
//		float median_displacements = getMedianAndDoPartition(displacements);
//		if(median_displacements > 10){
//			return BAD_PREDICT;
//		}
//		else {


//			StateType measureMov;

//			cv::Rect2f lmedPos;
//			lmedPos = LMEBox(gme_forward_trans, m_latestPos);
//			measureMov.dx = newPos.x + newPos.width /2.0 - lmedPos.x - lmedPos.width /2.0;
//			measureMov.dy = newPos.y + newPos.height/2.0 - lmedPos.y - lmedPos.height/2.0;

////			measureMov.dx = newPos.x + newPos.width /2.0 - m_latestPos.x - m_latestPos.width /2.0;
////			measureMov.dy = newPos.y + newPos.height/2.0 - m_latestPos.y - m_latestPos.height/2.0;

//			measurement.at<float>(0, 0) = measureMov.dx;
//			measurement.at<float>(1, 0) = measureMov.dy;
//			kf.correct(measurement);

//			return PREDICT_SUCESS;
//		}

//		m_latestPos = newPos;
//		m_old_points = predict_good_pts;
//		break;
//	}
	case TRACK_TYPE_KAL:
	{
		cv::Mat p = kf.predict();
		StateType predictionMovement;
		predictionMovement.dx = p.at<float>(0, 0);
		predictionMovement.dy = p.at<float>(1, 0);

		// update latestSize using gme
		float scale = (float)sqrt( gme_forward_trans.at<double>(0,0) * gme_forward_trans.at<double>(0,0)
								 + gme_forward_trans.at<double>(0,1) * gme_forward_trans.at<double>(0,1) );

		m_latestPos.x += predictionMovement.dx;
		m_latestPos.y += predictionMovement.dy;
		m_latestPos.width  = (int)(scale * m_latestPos.width + 0.5);
		m_latestPos.height = (int)(scale * m_latestPos.height + 0.5);
		update_lastest_position(m_latestPos);
		m_old_points.clear();
		if (predictionMovement.dx > 500 || predictionMovement.dx < -500 ||
			predictionMovement.dy > 500 || predictionMovement.dy < -500)
			return BAD_PREDICT;
#ifdef DEBUG
//		std::cout << "predict kalman - dx: " << predictionMovement.dx << ", dy: "
//				  << predictionMovement.dy << std::endl;
#endif
		return PREDICT_SUCESS;
		break;
	}
	default:
		break;
	}
	return BAD_PREDICT;
}

void DTracker::update(std::vector<cv::Point2f> newPts, cv::Rect trueObj, cv::Mat &gme_forward_trans)
{
	_update();
	m_type = getTrackType(newPts);
	switch (getTrackType(newPts)) {
	case TRACK_TYPE_OPT:
	{

		break;
	}
//	case TRACK_TYPE_MED:
//	{

//		break;
//	}
	case TRACK_TYPE_KAL:
	{
		StateType measuredMov;
		cv::Rect2f lmedPos;
		lmedPos = LMEBox(gme_forward_trans, m_latestPos);
		measuredMov.dx = trueObj.x + trueObj.width / 2.0 - lmedPos.x - lmedPos.width / 2.0;
		measuredMov.dy = trueObj.y + trueObj.height/ 2.0 - lmedPos.y - lmedPos.height/ 2.0;
//		measuredMov.dx = trueObj.x + trueObj.width / 2.0 - m_latestPos.x - m_latestPos.width / 2.0;
//		measuredMov.dy = trueObj.y + trueObj.height/ 2.0 - m_latestPos.y - m_latestPos.height/ 2.0;
		measurement.at<float>(0, 0) = measuredMov.dx;
		measurement.at<float>(1, 0) = measuredMov.dy;
//		std::cout << "update kalman - dx: " << measuredMov.dx << ", dy: " << measuredMov.dy << std::endl;
		kf.correct(measurement);
		break;
	}
	default:
		break;
	}
	m_latestPos = trueObj;
	update_lastest_position(m_latestPos);
	m_old_points = newPts;
}

void DTracker::_init()
{
	m_time_since_update = 0;
	m_hits = 0;
	m_hit_streak = 0;
	m_age = 0;
	m_id = id_seed++;
}

void DTracker::_predict()
{
	m_age += 1;
	if (m_time_since_update > 0)
		m_hit_streak = 0;
	m_time_since_update += 1;
}

void DTracker::_update()
{
	m_time_since_update = 0;
	m_hits += 1;
	m_hit_streak += 1;
}

cv::Rect DTracker::get_lastest_position()
{
//	return m_latestPos;
	return m_smoothLastePos;
}

int DTracker::getTrackType(std::vector<cv::Point2f> &pts)
{
//	if (pts.size() > 5) return TRACK_TYPE_MED;
//	return TRACK_TYPE_OPT;
	if (pts.size() > 15) return TRACK_TYPE_OPT;
//	if (pts.size() > 5) return TRACK_TYPE_MED;
	return TRACK_TYPE_KAL;
}

void DTracker::update_lastest_position(cv::Rect _lastest)
{
    m_smoothLastePos.x -= LPF_Beta * (m_smoothLastePos.x - _lastest.x);
    m_smoothLastePos.y -= LPF_Beta * (m_smoothLastePos.y - _lastest.y);
//    m_smoothLastePos.x = _lastest.x;
//    m_smoothLastePos.y = _lastest.y;
	m_smoothLastePos.width  -= LPF_Beta * (m_smoothLastePos.width  - _lastest.width );
	m_smoothLastePos.height -= LPF_Beta * (m_smoothLastePos.height - _lastest.height);
}

