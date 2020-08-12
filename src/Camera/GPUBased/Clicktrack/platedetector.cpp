#include "platedetector.h"

PlateDetector::PlateDetector()
{

}

PlateDetector::~PlateDetector()
{
	delete roi_detector;
}

bool PlateDetector::detect_RGBA(image_t input, cv::Point clickPoint, int trackSize, bbox_t &box_to_track)
{
	bool status = false;
	getRegion(clickPoint, trackSize, cv::Size(input.w, input.h));
    if (is_inside(roi_detect, cv::Size(input.w, input.h)) == false)
		return false;

	std::vector<bbox_t> result_vec;
    result_vec = roi_detector->gpu_detect_roi_RGBA(input, roi_detect, 0.15f);
//	std::cout << "result_vec: " << result_vec.size() << std::endl;

	status = select_best_box_to_track(result_vec, box_to_track, clickPoint, trackSize, true);

	return status;
}

bool PlateDetector::detect_I420(image_t input, cv::Point clickPoint, int trackSize, bbox_t &box_to_track)
{
    bool status = false;
    getRegion(clickPoint, trackSize, cv::Size(input.w, input.h * 2 /3));
    if (is_inside(roi_detect, cv::Size(input.w, input.h * 2 /3)) == false)
        return false;

    std::vector<bbox_t> result_vec;
    result_vec = roi_detector->gpu_detect_roi_RGBA(input, roi_detect, 0.15f);
//	std::cout << "result_vec: " << result_vec.size() << std::endl;

    status = select_best_box_to_track(result_vec, box_to_track, clickPoint, trackSize, true);

    return status;
}

bool PlateDetector::detect_I420(image_t input, cv::Rect trackBox, bbox_t &box_to_track)
{
    bool status = false;
    getRegion(trackBox, cv::Size(input.w, input.h * 2 /3));
    if (is_inside(roi_detect, cv::Size(input.w, input.h * 2 /3)) == false)
        return false;

    std::vector<bbox_t> result_vec;
    result_vec = roi_detector->gpu_detect_roi_I420(input, roi_detect, 0.15f);
//    if (result_vec.size() > 0)
//        std::cout << "\nresult_vec: " << result_vec.size() << std::endl;

    status = select_best_box_to_track(result_vec, box_to_track, trackBox, false);

    return status;
}


bool PlateDetector::is_inside(cv::Rect rect, cv::Size imgSize)
{
    if (rect.x < 0) return false;
    if (rect.y < 0) return false;
    if (rect.x + rect.width >= imgSize.width ) return false;
    if (rect.y + rect.height >= imgSize.height ) return false;
    return true;
}

bool PlateDetector::eliminate_box(std::vector<bbox_t>& boxs, int trackSize)
{
	if (boxs.size() == 0)
	{
		return 0;
	}

	int w, h;
	std::vector<bbox_t>::iterator it = boxs.begin();

	while (it != boxs.end())
	{
		w = it->w;
		h = it->h;

		// use area
		int object_area = w*h;
		int R = trackSize*trackSize;

		if (object_area < R/9 || object_area > R*4)
//		if (w < trackSize * 0.5 || w > trackSize * 2 || h < trackSize * 0.5 || h > trackSize * 2)
		{
			it = boxs.erase(it);
		}
		else
		{
			++it;
		}
	}

	if (boxs.size() == 0)
	{
		return 0;
	}

	return 1;
}

bool PlateDetector::eliminate_box(std::vector<bbox_t>& boxs, cv::Rect trackBox)
{
    if (boxs.size() == 0)
    {
        return 0;
    }

    int w, h;
    std::vector<bbox_t>::iterator it = boxs.begin();

    while (it != boxs.end())
    {
        w = it->w;
        h = it->h;

        // use area
        int object_area = w*h;
        int R = trackBox.width * trackBox.height;

        if (object_area < R/9 || object_area > R*4)
//		if (w < trackSize * 0.5 || w > trackSize * 2 || h < trackSize * 0.5 || h > trackSize * 2)
        {
            it = boxs.erase(it);
        }
        else
        {
            ++it;
        }
    }

    if (boxs.size() == 0)
    {
        return 0;
    }

    return 1;
}



bool PlateDetector::select_best_box_to_track(std::vector<bbox_t>& boxs, bbox_t& best_box, cv::Point clickPoint, const int trackSize, bool filter)
{
	if (boxs.size() == 0)
	{
		return false;
	}

	if (filter)
	{
		if (!eliminate_box(boxs, trackSize))
		{
			return false;
		}

		if (boxs.size() == 0)
		{
			return false;
		}
	}

	best_box = boxs[0];

	if (boxs.size() == 1)
	{
		return true;
	}

	bbox_t box;
	int distance = INT_MAX;
	int x_center, y_center;
	int w, h;
	uint idx_min = -1;

	for (uint i = 0; i < boxs.size(); i++)
	{
		box = boxs[i];
		w = box.w;
		h = box.h;
		x_center = box.x + w / 2;
		y_center = box.y + h / 2;
		int cur_distance = (clickPoint.x - x_center) * (clickPoint.x - x_center);
		cur_distance += (clickPoint.y - y_center) * (clickPoint.y - y_center);

		if (cur_distance <= distance)
		{
			distance = cur_distance;
			idx_min = i;
		}
	}

	if (idx_min >= 0 && distance <= 2*trackSize*trackSize)
	{
		best_box = boxs[idx_min];
		return true;
	}

	return false;
}

bool PlateDetector::select_best_box_to_track(std::vector<bbox_t>& boxs, bbox_t& best_box, cv::Rect trackBox, bool filter)
{
    if (boxs.size() == 0)
    {
        return false;
    }

    if (filter)
    {
        if (!eliminate_box(boxs, trackBox))
        {
            return false;
        }

        if (boxs.size() == 0)
        {
            return false;
        }
    }

    best_box = boxs[0];

    if (boxs.size() == 1)
    {
        return true;
    }

    bbox_t box;
    int distance = INT_MAX;
    int x_center, y_center;
    int w, h;
    uint idx_min = -1;

    for (uint i = 0; i < boxs.size(); i++)
    {
        box = boxs[i];
        w = box.w;
        h = box.h;
        x_center = box.x + w / 2;
        y_center = box.y + h / 2;
        cv::Point clickCenter;
        clickCenter.x = trackBox.x + trackBox.width  / 2.0;
        clickCenter.y = trackBox.y + trackBox.height / 2.0;
        int cur_distance = (clickCenter.x - x_center) * (clickCenter.x - x_center);
        cur_distance += (clickCenter.y - y_center) * (clickCenter.y - y_center);

        if (cur_distance <= distance)
        {
            distance = cur_distance;
            idx_min = i;
        }
    }

    if (idx_min >= 0 && distance <= 2*trackBox.width*trackBox.height)
    {
        best_box = boxs[idx_min];
        return true;
    }

    return false;
}


void PlateDetector::setDetector(Detector *_detector)
{
    roi_detector = _detector;
}

void PlateDetector::getRegion(cv::Point clickPoint, int trackSize, cv::Size frameSize)
{
	// we don't use trackSize here anymore; fix roi.size == 512x512
	int top, left, right, bottom;
	left = MAX(0, clickPoint.x - 256);
	top = MAX(0, clickPoint.y - 256);
	right = MIN(clickPoint.x + 256, frameSize.width - 1);
	bottom = MIN(clickPoint.y + 256, frameSize.height - 1);
	roi_detect.x = left;
	roi_detect.y = top;
	roi_detect.width = right - left;
	roi_detect.height = bottom - top;
	return;
}

void PlateDetector::getRegion(cv::Rect trackBox, cv::Size frameSize)
{
    // we don't use trackSize here anymore; fix roi.size == 512x512
    int top, left, right, bottom;
    cv::Point center;
    center.x = trackBox.x + trackBox.width  / 2.0;
    center.y = trackBox.y + trackBox.height / 2.0;
    left = MAX(0, center.x - 256);
    top = MAX(0, center.y - 256);
    right = MIN(center.x + 256, frameSize.width - 1);
    bottom = MIN(center.y + 256, frameSize.height - 1);
    roi_detect.x = left;
    roi_detect.y = top;
    roi_detect.width = right - left;
    roi_detect.height = bottom - top;
    return;
}

