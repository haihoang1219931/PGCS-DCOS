#include "PlateOCR.h"
#include <numeric>

using namespace std;

void PlateOCR::contrastEnhance(cv::Mat &scr, cv::Mat &dst, int dist)
{
	cv::Mat smooth;
	cv::GaussianBlur(scr, smooth, cv::Size(0, 0), 3);
	int a, b;
	int val, smt;
	for(int x = 0; x < scr.cols; x++)
		for(int y = 0 ; y < scr.rows; y++)
		{
			val =(int) scr.at<uchar>(y, x);
			smt = (int) smooth.at<uchar>(y, x);
			if((val - smt) > dist) smt = smt + (val-smt) * 0.5;
			smt = smt < 0.5 * dist ? 0.5 * dist : smt;
			b = smt + 0.5 * dist;
			b = b > 255 ? 255 : b;
			a = b - dist;
			a = a < 0 ? 0 : a;
			if(val >= a && val <= b )
			{
				dst.at<uchar>(y,x) = (int)(((val -a) / (0.5* dist)) * 255);
			}
			else if (val < a )
			{
				dst.at<uchar>(y,x) = 0;
			}
			else if (val > b)
			{
				dst.at<uchar>(y,x) = 255;
			}
		}
}

cv::Mat PlateOCR::deskewImage(cv::Mat image)
{
	cv::Mat display= image.clone();
	std::vector<cv::Mat> chanels;
	cv::Mat hsv, gray;
	if(image.channels() > 1)
	{
		cv::cvtColor(image, hsv, CV_RGB2HSV);
		cv::split(hsv, chanels);
		gray = chanels[2];
	}
	else gray = image.clone();
	cv::Mat enhanced(gray.size(), CV_8UC1);
	contrastEnhance(gray, enhanced);

	cv::Mat thresh1;
	cv::threshold(enhanced, thresh1, 90, 255, cv::THRESH_OTSU);
	//    cv::imshow("thresh", ~thresh1);
	//    cv::waitKey(0);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	//    thresh1 = ~thresh1;
	cv::findContours(thresh1, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
	cv::Mat contourMap = cv::Mat::zeros(cv::Size(thresh1.cols, thresh1.rows), CV_8U);
	int bigestContourIdx = -1;
	float bigestContourArea = 0;
	cv::Rect ctBox;
	float ctArea;
	std::vector<std::vector<cv::Point>> charcontours;
	for(int i = 0; i < contours.size(); i++)
	{
		ctArea = cv::contourArea(contours[i]);
		if(ctArea < 30) {continue;}
		ctBox = cv::boundingRect(contours[i]);
		//        cv::Mat maskroi(contourMap, ctBox);
		//        if(ctBox.area() > 30 && ctBox.width > 2 && ctBox.height > 8)// && r > 0.25)
		//        {
		//            charcontours.push_back(contours[i]);
		////            cv::drawContours(contourMap, contours, i, cv::Scalar(255, 255, 255), 1);
		//        }
		if(ctArea > bigestContourArea)
		{
			bigestContourArea = ctArea;
			bigestContourIdx = i;         }
	}
	cv::Mat plgray;
	if(bigestContourIdx > 0)
	{

		//    if(charcontours.size() < 3)
		//    {
		//        cout << " Can not recog " << endl;
		//    }
		cv::RotatedRect boundingBox = cv::minAreaRect(contours[bigestContourIdx]);
		//    cv::Point2f corners[4];
		//    boundingBox.points(corners);
		//    cv::line(display, corners[0], corners[1], cv::Scalar(255, 255, 255));
		//    cv::line(display, corners[1], corners[2], cv::Scalar(255, 255, 255));
		//    cv::line(display, corners[2], corners[3], cv::Scalar(255, 255, 255));
		//    cv::line(display, corners[3], corners[0], cv::Scalar(255, 255, 255));
		float angle = boundingBox.angle;
		if(angle <= -45.0 && angle >=-90.0)
		{
			angle = 90.0 + angle;
		}
		if(angle >= 90.0)
		{
			angle = angle - 90.0;
		}
		if(angle >= 45.0 && angle <= 90.0)
		{
			angle = 90.0 - angle;
		}

		if(abs(angle) > 4)
		{
			cv::Point2f center = cv::Point2f((float)thresh1.cols / 2.0, (float)thresh1.rows / 2.0);
			cv::Mat R = cv::getRotationMatrix2D( center, angle, 1.0 );
			cv::warpAffine(gray, gray, R, thresh1.size(), cv::INTER_CUBIC );
			//            cout << " size of sub [ W - H ]" << (int)boundingBox.size.width << "-" <<
			//                    boundingBox.size.height << endl;
			float ratio = boundingBox.size.width / boundingBox.size.height;
			if (ratio < 0.8)
			{
				cv::getRectSubPix(gray, cv::Size((int)boundingBox.size.height, (int)boundingBox.size.width),
								  boundingBox.center, plgray);
			}else
			{
				cv::getRectSubPix(gray, cv::Size((int)boundingBox.size.width, (int)boundingBox.size.height),
								  boundingBox.center, plgray);}
		}
		else plgray = gray.clone();
	}
	else plgray = gray.clone();
	//    string nm = imagename + "_thr.jpg";
	//    cv::imwrite(nm, plgray);

	return plgray;

}

std::string gen_random(const int len)
{
    char s[len];
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGH";

    for (int i = 0; i < len - 1; i++) {
        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    s[len - 1] = 0;
    return std::string(s);
}

std::vector<uint> PlateOCR::sort_indexes(std::vector<bbox_t> & track_vec) {
	std::vector<int> ids {4, 5, 6, 8 , 9, 10};
	std::vector<uint> idx(track_vec.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(),
			  [&track_vec, &ids](size_t i1, size_t i2) {
		if (std::find(ids.begin(), ids.end(), track_vec[i1].obj_id) == ids.end())
			return false;
		if (std::find(ids.begin(), ids.end(), track_vec[i2].obj_id) == ids.end())
			return true;
		int w1 = track_vec[i1].w * track_vec[i1].h;
		int w2 = track_vec[i2].w * track_vec[i2].h;
		return  w1 > w2;});
	return idx;
}

std::vector<bbox_t> PlateOCR::getPlateBoxes(const image_t& frame, const cv::Rect& _roi)
{
	cv::Rect roi = _roi;
	int right = MIN(frame.w       - 1, _roi.x + _roi.width);
	int bottom= MIN(frame.h * 2/3 - 1, _roi.y + _roi.height);
    roi.x = MAX(roi.x, 0);
    roi.y = MAX(roi.y, 0);
	roi.width = right - roi.x;
	roi.height = bottom - roi.y;
//    std::cout << "roi: " << roi << " | " << _roi << std::endl;
	return m_plate_detector->gpu_detect_roi_I420(frame, roi, 0.2f, false);
}

std::string PlateOCR::getPlateString(const image_t& frame, const cv::Mat &cpu_gray_frame, const cv::Mat &cpu_bgr_frame, const bbox_t& box)
{
//    printf("%s Line[%d]\r\n",__func__,__LINE__);
//	if (m_maxPlateDetect-- < 0) return std::string();
//    printf("%s Line[%d]\r\n",__func__,__LINE__);
//    cv::Rect searchRoi(0, 72, 1280, 576);
    cv::Rect roi(
                static_cast<int>(box.x),
                static_cast<int>(box.y),
                static_cast<int>(box.w),
                static_cast<int>(box.h));
//    if ((box.w * box.h) < 12000 /*&& (roi & searchRoi) != roi*/)
//		return std::string();
//    printf("%s Line[%d]\r\n",__func__,__LINE__);
    auto plates_result_vec = getPlateBoxes(frame, roi);

//    std::cout << "plates_result_vec len: " << plates_result_vec.size() << std::endl;
    for(auto i: plates_result_vec)
    {
        // loai bo xe co
//		if(i.w > 40 && i.h > 20 && i.w * i.h > 1200 && i.w > i.h)
		{
            std::cout << "possible plate" << std::endl;
			cv::Rect r(i.x, i.y, i.w, i.h);
            // Rectangle for checking type of plates: White or blue?
            cv::Rect p_rect(r.x >= 0 ? r.x : 0,
                            r.y >= 0 ? r.y : 0,
                            r.x + r.width < cpu_gray_frame.cols ? r.width : cpu_gray_frame.cols - r.x,
                            r.y + r.height < cpu_gray_frame.rows ? r.height : cpu_gray_frame.rows - r.y);
            cv::Mat bgr_plate = cpu_bgr_frame(p_rect).clone();
            double minVal, maxVal;
            cv::Mat float_plate;
            int blue = 0;
            int white = 0;
            int black = 0;
            cv::minMaxLoc(bgr_plate, &minVal, &maxVal);
            bgr_plate.convertTo(float_plate, CV_32FC3);

            for(uint r = 0; r < float_plate.rows; r++)
                for(uint c = 0; c  < float_plate.cols; c++)
                {
                    for(int d = 0; d < 3; d++)
                        float_plate.at<float>(r, 3 * c + d) = std::floor(255 * (float_plate.at<float>(r, 3 * c + d) - minVal) / (maxVal - minVal));
                    if(std::abs(float_plate.at<float>(r, 3 * c) - float_plate.at<float>(r, 3 * c + 1)) >= 50.f || std::abs(float_plate.at<float>(r, 3 * c) - float_plate.at<float>(r, 3 * c + 2)) >= 50.f)
                        blue++;
                    else
                    {
                        if(float_plate.at<float>(r, 3 * c) < 50.f && float_plate.at<float>(r, 3 * c + 1) < 50.f && float_plate.at<float>(r, 3 * c + 2) < 50.f)
                            black++;
                        else
                            white++;
                    }
                }

            float blue_rate = (float)blue / (float)(bgr_plate.rows * bgr_plate.cols - black);
            // End of checking
            r.x = MAX(r.x - 0.05 * r.width,  0);
            r.y = MAX(r.y - 0.05 * r.height, 0);
            r.width  = MIN(cpu_gray_frame.cols - r.x - 1, 1.1*r.width);
            r.height = MIN(cpu_gray_frame.rows - r.y - 1, 1.1*r.height);
//			r.width = r.x + 1.1 * r.width < frame.w ? (int)(1.1 * r.width) : frame.w - r.x;
//			r.height = r.y + 1.1 * r.height < frame.h ? (int)(1.1 * r.height) : frame.h - r.y;
//            std::cout << "R: " << r << std::endl;
//            assert(!cpu_gray_frame(r).empty());

            cv::Mat cpu_plateimage( cpu_gray_frame(r) );
			cv::Mat cpu_thresh_plate = deskewImage(cpu_plateimage);
//            cv::imwrite("/home/pgcs-01/Desktop/imgs/0.jpg", cpu_thresh_plate);

			// TODO: output string from cpu_thresh_plate
            // do something with m_OCR
            if(!cpu_thresh_plate.empty())
            {
                if(blue_rate >= 0.1f)
                {
                    cpu_thresh_plate = ~cpu_thresh_plate;
                }
                int plateType = i.obj_id;
                int sign = -1;
                std::vector<cv::Mat> chars = preprocess(cpu_thresh_plate, plateType, &sign);
                if(chars.size() > 6)
                {
                    std::string code = m_OCR->recognize(chars, sign);
                    printf("Code : %s\n", code.c_str());
                    int cc = 0;
                    for(uint l = 0; l < code.size(); l++)
                    {
                        if(code[l] != '_')
                            cc++;
                    }
//                    if(cc > 7)
                    {
                        return code;
                    }
                }
            }

//			return gen_random(8);

		}
    }
	return std::string();
}

PlateOCR::PlateOCR()
{
	m_maxPlateDetect = 10;
    m_logFile = FileController::get_day();
}

PlateOCR::~PlateOCR()
{
	delete m_plate_detector;
}

void PlateOCR::setPlateDetector(Detector * _plate_detector)
{
	m_plate_detector = _plate_detector;
}

void PlateOCR::setOCR(OCR* _OCR)
{
    m_OCR = _OCR;
}

//void PlateOCR::run(std::vector<bbox_t> & track_vec, const image_t &frame, const cv::Mat &cpu_gray_frame, const cv::Mat &cpu_bgr_frame, int max_info_read)
//{
//    // Define region for searching
//    cv::Rect searchRoi(0, 72, 1280, 576);
//	m_maxPlateDetect = max_info_read;
//    for (auto i : sort_indexes(track_vec)) {// Sort objects that are tracked
//        cv::Rect r(track_vec[i].x, track_vec[i].y, track_vec[i].w, track_vec[i].h);
//        if(std::find(wanted_class.begin(), wanted_class.end(), track_vec[i].obj_id) == wanted_class.end() || (r & searchRoi) == r)
//            continue;

//        unsigned int cur_track_id = track_vec[i].track_id;
//        if (!data.count(cur_track_id)) {
//            std::vector<std::string> current_data;
//            current_data.push_back(std::string());
//            // Do OCR
//            std::string plate = getPlateString(frame, cpu_gray_frame, cpu_bgr_frame, track_vec[i]);
//            if (!plate.empty()) {
//                current_data.push_back(plate);
//                // update database
//                data.insert(std::pair<int, std::vector<std::string>>(cur_track_id, current_data));
//            }
//            // return
//            track_vec[i].track_info.stringinfo = plate;
//        }
//		else {
//            // query the data at cur_track_id
//            auto cur_data = data.at(cur_track_id);
//            // check if final exists
//            if (!cur_data[0].empty())
//            {
//                track_vec[i].track_info.stringinfo = cur_data[0];
//            }
//            else {
//                if (cur_data.size() >= 4)
//                {
//                    // combine here
//                    // dau vao: cur_data[1,2,3]
//                    // dau ra final: cur_data[0]
//                    std::string temp = search_combine_plates(cur_data);

//                    // return
//                    cur_data[0] = temp;
//                    track_vec[i].track_info.stringinfo = cur_data[0];

//                    // update database
//                    data.at(cur_track_id) = cur_data;
//                }
//                else
//                {

//                    std::string plate = getPlateString(frame, cpu_gray_frame, cpu_bgr_frame, track_vec[i]);
////                    std::string time = FileController::get_time_stamp();
////                    std::string imgFile = "plates/"+ time+"_"+plate+".png";
////                    std::string lineLog = time + ";" + plate + ";" + imgFile;
//                    if (!plate.empty())
//                        cur_data.push_back(plate);
//                    track_vec[i].track_info.stringinfo = cur_data.back(); // the last detected plate
//                    // update database
//                    data.at(cur_track_id) = cur_data;
//                }
//            }
//		}
//	}
//}
void PlateOCR::run(std::vector<bbox_t> & track_vec, const image_t &frame, const cv::Mat &cpu_gray_frame, const cv::Mat &cpu_bgr_frame, int max_info_read){
    return;
    //    printf("OCR run\r\n");
    int count = 0;
    for (int i=0; i< track_vec.size(); i++) {
        // select object except bike pedestrian
        cv::Rect r(track_vec[i].x, track_vec[i].y, track_vec[i].w, track_vec[i].h);
        std::string plate = getPlateString(frame, cpu_gray_frame, cpu_bgr_frame, track_vec[i]);
        track_vec[i].track_info.stringinfo = plate;
//        std::cout << count  << "[" <<plate << "]" <<std::endl;
        count++;
    }
}
bool isUpperCase(const char c)
{
	if((int)c >= 65 && (int)c <= 90)
		return true;
	return false;
}

float PlateOCR::get_strings_correlation(const std::string& input, const std::string& truth)
{
/*	if (input.empty() || truth.empty()) return 0;
	int p1 = -1, p2 = -1;
	for(int i = 0; i < input.size(); i++)
		if (isUpperCase(input.at(i))) {
			p1 = i;
			break;
		}
	for(int i = 0; i < truth.size(); i++)
		if (isUpperCase(truth.at(i))) {
			p2 = i;
			break;
		}
	if (p1 < 0 || p2 < 0) return 0;
	int count = -1;
	for(int i = -3; i <= 5 ; i++) {
		char c1, c2;
		try { c1 = input.at(p1 + i);}
		catch (std::out_of_range e)
				 {continue;};
		try { c2 = truth.at(p2 + i);}
		catch (std::out_of_range e)
				 {continue;};

		if (c1 == c2) count++;
	}
	if (count < 0)
		return 0;
	else
        return float(count) / truth.size();*/
    std::string headCode,rearCode, hCode, rCode;
    int pos = 0;
    for(int i = 0; i < truth.size(); i++)
    {
        if(truth[i] == '/')
        {
            pos = i;
            break;
        }
        else if(truth[i] == '-')
        {
            pos = i;
        }
    }
    hCode = truth.substr(0, pos);
    rCode = truth.substr(pos + 1, truth.size() - pos - 1);
    pos = 0;
    for(int i = 0; i < input.size(); i++)
    {
        if(input[i] == '/')
        {
            pos = i;
            break;
        }
        else if(input[i] == '-')
        {
            pos = i;
        }
    }
    headCode = input.substr(0, pos);
    rearCode = input.substr(pos + 1, input.size() - pos - 1);
    return 0.f;
}

std::string PlateOCR::search_combine_plates(std::vector<std::string> cur_data)
{
    std::vector<std::vector<std::pair<char, int>>> codeTable;
    std::vector<std::pair<char, int>> initSym = {std::make_pair('_', 1)};
    for(int i = 0; i < 9; i++)
        codeTable.push_back(initSym);
    for(int i = 1; i < 4; i++)
    {
        // Locate position separating region code to number order.
        int pos = 0;
        for(int j = 0; j < cur_data[i].size(); j++)
        {
            if(cur_data[i][j] == '/')
            {
                pos = j;
                break;
            }
            if(cur_data[i][j] == '-')
                pos = j;
        }
        // Push region code to code table.
        for(int j = 0; j < pos; j++)
        {
            bool inserted = false;
            for(int k = 0; k < codeTable[j].size(); k++)
            {
                if(cur_data[i][j] == codeTable[j][k].first)
                {
                    codeTable[j][k].second++;
                    inserted = true;
                    break;
                }
            }
            if(!inserted)
            {
                codeTable[j].push_back(std::make_pair(cur_data[i][j], 1));
            }
        }
        // Push number order to code table.
        int ct = 4;
        for(int j = pos + 1; j < cur_data[i].size(); j++)
        {
            bool inserted = false;
            for(int k = 0; k < codeTable[ct].size(); k++)
            {
                if(cur_data[i][j] == codeTable[ct][k].first)
                {
                    codeTable[ct][k].second++;
                    inserted = true;
                    break;
                }
            }
            if(!inserted)
            {
                codeTable[ct].push_back(std::make_pair(cur_data[i][j], 1));
            }
            ct++;
        }
    }
    // Get final result
    std::string s("");
    for(int i = 0; i < 9; i++)
    {
        std::sort(codeTable[i].begin(), codeTable[i].end(), [](std::pair<char, int> a, std::pair<char, int> b)
        {
            return a.second > b.second;
        });
        bool inserted = false;
        for(int j = 0; j < codeTable[i].size(); j++)
        {
            if(codeTable[i][j].first != '_')
            {
                s += codeTable[i][j].first;
                inserted = true;
                break;
            }
        }
        if(!inserted)
        {
            s+= '_';
        }
    }

//                    cur_data[0] = s;
    string temp;
    for(int i = 0; i < 9; i++)
    {
        if(i == 3)
        {
            if(s[i] != '_')
            {
//                                cur_data[0] += s[i];
                temp += s[i];
            }
//                            cur_data[0] += '-';
            temp += '-';
        }
        else
        {
//                            cur_data[0] += s[i];
            temp += s[i];
        }
    }
    // acutal return
//                    track_vec[i].track_info.stringinfo = cur_data.back();
    // should be
    return temp;
}
