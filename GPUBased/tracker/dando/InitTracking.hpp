#ifndef INIT_DETECTOR
#define INIT_DETECTOR
//----basic include

#include <iostream>

#include<string>
#include"opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "movdetection/mcdwrapper.hpp"
#include "HTrack/saliency.h"
#include"image_utils.hpp"
using namespace ImageUtility;

/* max value of variance in a patch to verify that there is a chance
that an object exists in the patch*/

#define MAX_VARIANCE 0.13

/* calculate ratio between area of detected object and searching window area
 if this ratio is lower than MIN_OBJ_RATIO, discard object found */

#define MIN_OBJ_RATIO 0.0

class InitTracking
{
    public:
        bool foundObject;
        cv::Mat original_img;
        cv::Mat processed_img;
        cv::Point clicked_center;
        int default_size;
        cv::Rect object_position;
        //    bool start_finding_object ;
        MCDWrapper movDetector;
    public:
        InitTracking();
        ~InitTracking();
        void Init(cv::Mat original_img, cv::Point clicked_center, int default_size);
        void UpdateMovingObject(cv::Mat current_image);
        void Run();
    private:
        IplImage ProcessMovObjImg;
        IplImage *ProcessMovObjBuffer;
        IplImage *InitProcessMovObjBuffer;
        bool ProcessMovingObjInit;
        bool checkPatchDetail(cv::Mat &patch);
};

#endif // INIT_DETECTOR

