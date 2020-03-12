#include "InitTracking.hpp"


// set all internal flag of the class
InitTracking::InitTracking()
{
    ProcessMovingObjInit =  false;
    //    start_finding_object = false;
}
/**
 * @brief InitTracking::InitTracking
 * @param original_img: grayscale image of current frame of the sequence
 * @param processed_img: current frame of the sequence after several preprocessing
 * @param clicked_center: click point defined by user
 * @param default_size: default size of searching window
 */
void InitTracking::Init(cv::Mat original_img, cv::Point clicked_center, int default_size)
{
    if (original_img.channels() > 1) {
        cv::cvtColor(original_img, this->original_img, CV_BGR2GRAY);
    } else {
        this->original_img = original_img;
    }
    this->processed_img = this->original_img;
    this->default_size = default_size;
    this->clicked_center = clicked_center;
    foundObject = false;
}
InitTracking::~InitTracking()
{
}
/**
 * @brief InitTracking::UpdateMovingObject
 * This function is used to continously update moving object in image in sequence
 * Final result will be stored in attribute processed_img of the class
 * @param current_image
 */

void InitTracking::UpdateMovingObject(cv::Mat current_image)
{
    ProcessMovObjImg = current_image;
    InitProcessMovObjBuffer = &ProcessMovObjImg;

    if (!ProcessMovingObjInit) {
        ProcessMovObjBuffer = cvCreateImage(current_image.size(), IPL_DEPTH_8U, 3);
        cvCopy(InitProcessMovObjBuffer, ProcessMovObjBuffer, 0);
        // init image for moving object detection
        movDetector.Init(ProcessMovObjBuffer);
        ProcessMovingObjInit = true;
    } else {
        cvCopy(InitProcessMovObjBuffer, ProcessMovObjBuffer, 0);
        // start finding moving object
        movDetector.Run();
        cv::Mat tmp(movDetector.detect_img->height, movDetector.detect_img->width, CV_8UC1,
                    (void *)movDetector.detect_img->imageData);
        cv::dilate(tmp, tmp, cv::Mat(), cv::Point(-1, -1), 5);
        cv::erode(tmp, processed_img, cv::Mat(), cv::Point(-1, -1), 5);
    }
}
/**
 * @brief InitTracking::checkPatchDetail
 * This function is used to check whether an object exists in a patch or not
 * @param patch
 * @return
 */
bool InitTracking::checkPatchDetail(cv::Mat &patch)
{
    cv::Scalar  meanVal, stdVal;
    //    cv::cuda::GpuMat gpuImg;
    //    gpuImg.upload(patch);
    //        cv::cuda::meanStdDev(gpuImg, meanVal, stdVal);
    cv::meanStdDev(patch, meanVal, stdVal);
    double variance = stdVal[0] / meanVal[0];

    if (variance < MAX_VARIANCE) {
        return false;
    } else {
        return true;
    }
}
/**
 * @brief InitTracking::Run
 * This function is used to detect object in a patch
 * Location of object will be store in attribute object_position of the class
 */

void InitTracking::Run()
{
    //==========truncate selected bounding box when it is outside the frame ===================
    int     top = (int)clicked_center.y - default_size,
            left = (int)clicked_center.x - default_size,
            bot = (int)clicked_center.y + default_size,
            right = (int)clicked_center.x + default_size;

    if (top < 0) {
        top = 0;
    }

    //
    if (left < 0) {
        left = 0;
    }

    if (bot >= (int)processed_img.size().height) {
        bot = (int)processed_img.size().height - 1;
    }

    if (right >= (int)processed_img.size().width) {
        right = (int)processed_img.size().width - 1;
    }

    int width = right - left,
        height = bot - top;
    //----------- variable definition-----------------------------------------------------
    std::vector<std::vector<cv::Point>> contour;
    std::vector<cv::Point> ret_contour;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat gray, thresh_gray;
    // calculate ratio between area std::cout << "=============> 2\n";of detected object and searching window
    // if this ratio is lower than MIN_OBJ_RATIO, discard object found
    float object_ratio = 1;
    //----------------crop region of interest around clicked point by user----------------
    cv::Rect potential_area = cv::Rect(left, top, width, height);
    //    std::cout << "potential area" << potential_area << std::endl;
    //    std::cout << "original_img size:" << original_img.size() << std::endl;
    cv::Mat crop = original_img(potential_area);
    //    printf("3----------\n");
    /*if (checkPatchDetail(crop))*/ {
        cv::Mat mask = cv::Mat::zeros(processed_img.size(), CV_8UC1);
        mask(potential_area).setTo(255);
        processed_img &= mask;
        cv::findContours(processed_img, contour, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        contour.clear();
        cv::findContours(processed_img, contour, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        double max_contour_area = 0 ;

        if (contour.size() < 70 && contour.size() > 0) {
            for (unsigned long i = 0; i < contour.size(); i++) {
                if (cv::contourArea(contour[i]) > max_contour_area) {
                    ret_contour = contour[i];
                    max_contour_area = cv::contourArea(contour[i]);
                }
            }

            object_ratio = max_contour_area / default_size / default_size;
            object_position = cv::boundingRect(ret_contour);
            foundObject = true;
        }

        if ((contour.size() == 0 || object_ratio < MIN_OBJ_RATIO)) {
            object_ratio = 1;
            //            object_position = saliency(crop);
            cv::Mat clone = original_img.clone();
            //            clone = contrastEnhance(clone, 0.5);

            if (clone.channels() == 3) {
                cv::cvtColor(clone, gray, CV_BGR2GRAY);
            } else {
                clone.copyTo(gray);
            }

            double minVal, maxVal;
            cv::minMaxLoc(gray, &minVal, &maxVal);
            //            if (maxVal < 200)
            //                maxVal = 200;
            cv::threshold(gray, thresh_gray, maxVal - 30, maxVal + 10, CV_THRESH_BINARY);
            //            cv::adaptiveThreshold(gray, thresh_gray, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 15, -5);
            thresh_gray &= mask;
            contour.clear();
            cv::findContours(thresh_gray, contour, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

            if (contour.size() < 30) {
                //                printf("number of contour is %d\n", contour.size());
                max_contour_area = 0;

                for (unsigned long i = 0; i < contour.size(); i++) {
                    if (cv::contourArea(contour[i]) > max_contour_area) {
                        ret_contour = contour[i];
                        max_contour_area = cv::contourArea(contour[i]);
                    }
                }

                //        printf("max contour area is %f\n", max_contour_area);
                object_position = cv::boundingRect(ret_contour);
                foundObject = true;
            }
        }

        //        std::cout << "Object position is " << object_position << std::endl;
    }
}
