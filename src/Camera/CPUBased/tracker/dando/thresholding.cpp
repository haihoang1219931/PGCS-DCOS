#include "thresholding.hpp"

ThresholdingTracker::ThresholdingTracker()
{
    found_contour = true;
    m_trackInited = false;
    m_running = false;
}
ThresholdingTracker::~ThresholdingTracker()
{
}
bool ThresholdingTracker::checkPatchDetail(cv::Mat &patch)
{
    cv::Scalar  meanVal, stdVal;
    //    cv::cuda::GpuMat gpuImg;
    //    gpuImg.upload(patch);
    //        cv::cuda::meanStdDev(gpuImg, meanVal, stdVal);
    cv::meanStdDev(patch, meanVal, stdVal);
    double variance = stdVal[0] / meanVal[0];

    if (variance < max_variance) {
        return false;
    } else {
        return true;
    }
}
cv::Mat ThresholdingTracker::contrastEnhance(cv::Mat image, float weightingParam)
{
    cv::Mat intensity(image.size(), CV_32F);
    cv::Mat hsvImg, pdf;
    std::vector<cv::Mat> hsv;

    //================convert to HSV to get value channel============================
    if (image.channels() > 1) {
        image.convertTo(image, CV_32F);
        image = image / 255.;
        cv::cvtColor(image, hsvImg, CV_BGR2HSV);
        cv::split(hsvImg, hsv);
        hsv[2].copyTo(intensity);
        intensity *= 255.;
    } else {
        image.convertTo(image, CV_32F);
        image.copyTo(intensity);
    }

    //============get  pdf using calcHist============================================
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    cv::calcHist(&intensity, 1, 0, cv::Mat(), pdf, 1, &histSize, &histRange, true, false);
    pdf = pdf / (intensity.cols * intensity.rows);
    //===============gamma correction by applying weighting distribution function to calculate cdf========================
    double maxPdf, minPdf;
    cv::minMaxLoc(pdf, &minPdf, &maxPdf);
    cv::Mat cdf(pdf.rows, pdf.cols, pdf.type());

    for (int i = 0; i < pdf.rows; i++) {
        pdf.at<float>(i, 0) = maxPdf * std::pow((float)(pdf.at<float>(i, 0) - minPdf) / (maxPdf - minPdf), (float)weightingParam);

        if (i == 0) {
            cdf.at<float>(i, 0) = pdf.at<float>(i, 0);
        } else {
            cdf.at<float>(i, 0) = cdf.at<float>(i - 1, 0) + pdf.at<float>(i, 0);
        }
    }

    cdf = cdf / cdf.at<float>(cdf.rows - 1, 0);
    cv::Mat result = image.clone();
    int width = result.cols;
    int height = result.rows;
    float *data = (float *)intensity.data;
    auto fillData = [&](const cv::Range & r) {
        for (size_t i = r.start; i != r.end; i++) {
            int value = data[i];
            data[i] = 255 * std::pow((float)(value / 255.), (1 - cdf.at<float>(value, 0)));
        }
    };
    auto start = std::chrono::high_resolution_clock::now();
//    cv::parallel_for_(cv::Range(0, width * height), fillData);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time of enhance elapsed: " << elapsed.count() << std::endl;

    if (image.channels() > 1) {
        intensity /= 255.;
        result /= 255.;
        cv::cvtColor(result, result, CV_BGR2HSV);
        cv::split(result, hsv);
        hsv[2] = intensity;
        cv::merge(hsv, result);
        cv::cvtColor(result, result, CV_HSV2BGR);
        result *= 255.;
        result.convertTo(result, CV_8UC3);
    } else {
        intensity.copyTo(result);
        result.convertTo(result, CV_8UC1);
    }

    return result;
}

void ThresholdingTracker::initTrack(cv::Mat &_image, cv::Rect _selRoi)
{
    if ((_selRoi.width > 0) && (_selRoi.height > 0)) {
        selectedRoi = _selRoi;
        m_trackInited = true;
        motion_detector.setDefaultModel();
        currShift.x = currShift.y = 0;
    }
}

void ThresholdingTracker::performTrack(cv::Mat &_image)
{
    auto start = std::chrono::high_resolution_clock::now();
    m_running = true;
    //    printf("-----------begin the loop-----------------\n");

    //-------------convert to grayscale--------------------------------------
    if (_image.channels() == 3) {
        cv::cvtColor(_image, gray_frame, CV_BGR2GRAY);
    } else {
        _image.copyTo(gray_frame);
    }

    cv::Mat patch;
    //------------------limit crop image just in case bbox is out of range of the image-------------------
    cv::Rect rect_frame = cv::Rect(0, 0, gray_frame.size().width, gray_frame.size().height);
    //    std::cout << "Size of selectedRoi before limit is " << selectedRoi << std::endl;
    cv::Rect search_area = cv::Rect(selectedRoi.x - 30, selectedRoi.y - 30, 60 + selectedRoi.width, 60 + selectedRoi.height);

    if ((selectedRoi & rect_frame).area() > 0) {
        if (search_area.x < 0) {
            search_area.width = search_area.width + search_area.x;

            if (search_area.y < 0) {
                search_area.height = search_area.height + search_area.y;
                search_area.y = 0;
            } else if (search_area.br().y > gray_frame.size().height) {
                search_area.height -= (search_area.br().y - gray_frame.size().height);
            }

            search_area.x = 0;
        } else if (search_area.br().x > gray_frame.size().width) {
            search_area.width -= (search_area.br().x - gray_frame.size().width);

            if (search_area.y < 0) {
                search_area.height += search_area.y;
                search_area.y = 0;
            } else if (search_area.br().y > gray_frame.size().height) {
                search_area.height -= (search_area.br().y - gray_frame.size().height);
            }
        } else if (search_area.y < 0) {
            search_area.height += search_area.y;
            search_area.y = 0;
        } else if (search_area.br().y > gray_frame.size().height) {
            search_area.height -= (search_area.br().y - gray_frame.size().height);
        }
    } else {
        printf("Jump in this stupid thing \n");
        selectedRoi = cv::Rect(0, 0, 0, 0);
        search_area = cv::Rect(0, 0, 0, 0);
    }

    patch = gray_frame(search_area);
    //--------------------------------------------------------------------------------
    //-------------------get maxVal intensity in the patch----------------------------
    //--------------------------------------------------------------------------------
    /*if (!isTextureLess(patch))*/ {
        double minVal, maxVal;
        cv::minMaxLoc(patch, &minVal, &maxVal);
        cv::threshold(gray_frame, thresh_frame, maxVal - 20, maxVal + 10, CV_THRESH_BINARY);
        prevShift = currShift;
        //!! ===============================finding all contours which might be object's==========================
        //!
        object_center = cv::Point2f(selectedRoi.x + selectedRoi.width / 2, selectedRoi.y + selectedRoi.height / 2);
        motion_detector.setState(object_center.x, object_center.y, currShift.x, currShift.y);
        contour.clear();
        cv::findContours(thresh_frame, contour, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
        cv::Mat drawer = cv::Mat::zeros(_image.size(), CV_8UC1);
        cv::Mat draw_result_contour = cv::Mat ::zeros(_image.size(), CV_8UC1);
        cv::Mat draw_hull_result = cv::Mat ::zeros(_image.size(), CV_8UC1);
        //---------------------------------------------------------------------------
        //--------------bunch of variables for later use------------------------------
        //--------------------------------------------------------------------------
        std::vector<float> contour_length(contour.size());
        std::vector<cv::Moments> moment(contour.size());
        std::vector<cv::Point2f> centroid(contour.size());
        std::vector<double> distance(contour.size());
        std::vector<std::vector<cv::Point>> convexHullShape(contour.size());
        std::vector<double> circularity(contour.size());
        contour_length.clear();
        moment.clear();
        centroid.clear();
        distance.clear();
        convexHullShape.clear();
        circularity.clear();
        float maxLength = 0;
        double minDist = 1000000;
        unsigned long minIdx;
        unsigned long maxIdx;
        cv::Point2f center_potential_contour;
        float radius;
        float differRatio = 100;
        float SizeRatio;
        found_contour = false;
        float prevLen = sqrtf(prevShift.x * prevShift.x + prevShift.y * prevShift.y);
        //---------------------------------------------------------------------------
        //---------------------------------------------------------------------------
        motion_detector.predictMotion(&currShift.x, &currShift.y);
        motion_detector.correctModel(currShift.x, currShift.y);
        cv::Point estimate_position;
        estimate_position = motion_detector.get_estimate_position();
        cv::Mat rgb = _image.clone();
        cv::cvtColor(rgb, rgb, CV_GRAY2BGR);
        cv::circle(rgb, object_center, 3, cv::Scalar(255, 0, 255));

        for (size_t i = 0; i < contour.size(); i++) {
            if (hierarchy[i][2] < 0) {
                contour_length[i] = cv::arcLength(contour[i], false);
            } else {
                contour_length[i] = cv::arcLength(contour[i], true);
            }

            cv::minEnclosingCircle(contour[i], center_potential_contour, radius);
            //                printf("center_potential_contour is (%f, %f)\n", center_potential_contour.x, center_potential_contour.y);
            currShift.x = center_potential_contour.x - estimate_position.x;
            currShift.y = center_potential_contour.y - estimate_position.y;
            float currLen = sqrtf(currShift.x * currShift.x + currShift.y * currShift.y);

            if (currLen < 45 && radius < MAX_RADIUS) {
                SizeRatio = contour_length[i] / object_perimeter;
                cv::drawContours(drawer, contour, i, cv::Scalar(255, 0, 0));
                bool check_perimeter = 0;

                if (contour_length[i] >= maxLength) {
                    if (object_perimeter > 5000)
                        check_perimeter = 1;
                    else if (SizeRatio > 0.25)
                        check_perimeter = 1;

                    if (SizeRatio < 6 && check_perimeter) {
                        /*if (fabs(SizeRatio - 1) < differRatio)*/ {
                            //                            differRatio = fabs(SizeRatio - 1);
                            cv::drawContours(draw_result_contour, contour, i, cv::Scalar(255, 0, 0), 1);
                            cv::convexHull(contour[i], convexHullShape[i]);
                            found_contour = true;
                            maxIdx = i;
                            maxLength = contour_length[i];
                        }
                    }
                }
            }
        }

        //!! ====================Update feature of found object========================================
        //!
        //!

        if (found_contour) {
            // ================show found object propery===================================
            trackLostCnt = 0;
            object_perimeter = contour_length[maxIdx];
            cv::minEnclosingCircle(contour[maxIdx], center_potential_contour, radius);
            cv::Point2f potentialShift ;
            potentialShift.x = center_potential_contour.x - object_center.x;
            potentialShift.y = center_potential_contour.y - object_center.y;
            double potentialLen = sqrtf(potentialShift.x * potentialShift.x + potentialShift.y * potentialShift.y);
            cv::Rect temp;
            temp = cv::Rect(center_potential_contour.x - radius, center_potential_contour.y - radius, 2 * radius, 2 * radius);
            selectedRoi = temp;
            object_center = center_potential_contour;
            currShift = potentialShift;
            cv::circle(rgb, estimate_position, 3, cv::Scalar(255, 0, 255));

            if (selectedRoi.x > 10) {
                selectedRoi.x -= 10;
            } else {
                if (selectedRoi.x < 0) {
                    selectedRoi.width = selectedRoi.width + selectedRoi.x;
                    selectedRoi.x = 0;
                }
            }

            if (selectedRoi.y > 10) {
                selectedRoi.y -= 10;
            } else {
                if (selectedRoi.y < 0) {
                    selectedRoi.height = selectedRoi.height + selectedRoi.y;
                    selectedRoi.y = 0;
                }
            }

            if (selectedRoi.x > 10 && selectedRoi.y > 10) {
                selectedRoi.width += 20;
                selectedRoi.height += 20;
            }
        } else {
            printf("Jump in infeasible movement \n");
            trackLostCnt += 1;
            object_perimeter = 10000;

            if (trackLostCnt > 40) {
                trackLostCnt = 0;
                selectedRoi = cv::Rect(0, 0, 0, 0);
            }

            currShift = prevShift;
        }
    }
    m_running = false;
}
cv::Rect ThresholdingTracker::getPosition()
{
    return selectedRoi;
}
bool ThresholdingTracker::isInitialized()
{
    return m_trackInited;
}
bool ThresholdingTracker::isRunning()
{
    return m_running;
}
void ThresholdingTracker::resetTrack()
{
    m_trackInited = false;
}
