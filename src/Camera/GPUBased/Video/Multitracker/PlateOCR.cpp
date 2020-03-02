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
    int right = MIN(frame.w - 1, _roi.x + _roi.width);
    int bottom= MIN(frame.h - 1, _roi.y + _roi.height);
    roi.x = MAX(roi.x, 0);
    roi.y = MAX(roi.y, 0);
    roi.width = right - roi.x;
    roi.height = bottom - roi.y;
    return m_plate_detector->gpu_detect_roi(frame, roi, 0.2f, false);
}

std::string PlateOCR::getPlateString(const image_t& frame, const cv::cuda::GpuMat& gpu_rgba_frame, const bbox_t& box)
{
    if (m_maxPlateDetect-- < 0) return std::string();
    if ((box.w * box.h) < 12000)
        return std::string();
    cv::Rect roi(box.x, box.y, box.w, box.h);
    int right = MIN(frame.w - 1, box.x + box.w);
    int bottom= MIN(frame.h - 1, box.y + box.h);
    roi.x = MAX(roi.x, 0);
    roi.y = MAX(roi.y, 0);
    roi.width = right - roi.x;
    roi.height = bottom - roi.y;

    auto vehicle_result_vec = m_plate_detector->gpu_detect_roi(frame, roi, 0.2f, false);
    //    std::cout << "# of plates : " << vehicle_result_vec.size() << std::endl;

    for(auto i: vehicle_result_vec)
    {

        cv::Rect r(i.x, i.y, i.w, i.h);
        if((r & m_regA) == r && i.h > 10 && i.w > 35 && i.h < 1.5 * i.w)
        {
            r.x = r.x - 0.05 * r.width > 0 ? r.x - 0.05 * r.width : 0;
            r.y = r.y - 0.05 * r.height > 0 ? r.y - 0.05 * r.height : 0;
            r.width = r.x + 1.1 * r.width < frame.w ? (int)(1.1 * r.width) : frame.w - r.x;
            r.height = r.y + 1.1 * r.height < frame.h ? (int)(1.1 * r.height) : frame.h - r.y;

            cv::Mat cpu_thres_plate( gpu_rgba_frame(r));
//            cv::Mat cpu_thresh_plate = deskewImage(cpu_plateimage);
            cv::Mat plate;
            cv::cvtColor(cpu_thres_plate, plate, CV_BGRA2RGB);
            cv::cvtColor(plate, plate, CV_RGB2HSV);
            std::vector<cv::Mat> channels;
            cv::split(plate, channels);
            cv::Mat grayPlate = channels[2].clone();
            plate.release();
            channels.clear();
//            cv::imwrite("/home/pgcs-04/workspace/giapvn/imgs/" + std::to_string(m_cter) + ".png", grayPlate, {CV_IMWRITE_PNG_COMPRESSION, 0});
//            m_cter++;
            int plateType = i.obj_id;
            int sign = -1;
            std::vector<cv::Mat> chars;
            if(!grayPlate.empty())
                chars = preprocess(grayPlate, plateType, &sign);
            std::cout << "Size of chars : >>>>>>>>>>>>>>>>>> " << chars.size() << std::endl;
            if(chars.size() > 6)
            {
                std::string code = m_recognizer.recognize(chars, sign);
                int cc = 0;
                for(uint l = 0; l < code.size(); l++)
                {
                    if(code[l] != '_')
                        cc++;
                }
                if(cc > 7)
                {
                    return code;
                }
            }
            // TODO: output string from cpu_thresh_plate

        }
//        return gen_random(8);
    }
    return std::string();
}

PlateOCR::PlateOCR(std::string plate_cfg_file, std::string plate_weights_file)
{
    m_plate_detector = new Detector(plate_cfg_file, plate_weights_file);
    m_maxPlateDetect = 10;
}

PlateOCR::~PlateOCR()
{
    delete m_plate_detector;
}

void PlateOCR::run(std::vector<bbox_t> & track_vec, const image_t &frame, const cv::cuda::GpuMat &gpu_rgba_frame, int max_info_read)
{
    m_maxPlateDetect = max_info_read;
    for (auto i : sort_indexes(track_vec)) {
        if (data.count(track_vec[i].track_id)) {
            track_vec[i].track_info.stringinfo = data.at(track_vec[i].track_id);
        }
        else
        {
            std::string plate = getPlateString(frame, gpu_rgba_frame, track_vec[i]);
            if (plate.empty()) continue;
            track_vec[i].track_info.stringinfo = plate;
            data.insert(std::pair<int, std::string>(track_vec[i].track_id, plate));
        }
    }
}

