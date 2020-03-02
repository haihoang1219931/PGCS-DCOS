#ifndef UTILITIES_HPP
#define UTILITIES_HPP


#include <opencv2/opencv.hpp>
#include "HTrack/ffttools.hpp"


/** ===================================================
 *
 *              constants
 *
 * ====================================================
 */
#ifndef     PI
#define     PI                      3.141592654
#endif

#define     PADDING                 2.5f
#define     OUTPUT_SIGMA_FACTOR     0.1f
#define     LAMDA_                  0.0001f
#define     LAMDA_1                 1e-4

#define     SCALE_CHANGE_RATIO      1.12f
#define     SCALE_WEIGTH            0.94f

#define     OCCLUDED_THRESH_GAUSS   0.6f
#define     OCCLUDED_THRESH_POLY    600

#define     CELL_SIZE               4
#define     TEMPLATE_SIZE           96.0


#define     KERNEL_GAUSSIAN         0
#define     KERNEL_POLYNOMIAL       1

#define     FEATURE_HOG             0
#define     FEATURE_GRAY            1

#define     TRACK_INVISION          0
#define     TRACK_OCCLUDED          -1
#define     TRACK_LOST              -2

#define     TRACK_MAX_LOST_COUNT    30


//-------- Camera configuration
#define     IR_MIN_FOV              0.010472f
#define     IR_MAX_FOV              0.064577f
#define     IR_MAX_SIZE             720
#define     IR_MAX_OBJ_SIZE         200
#define     IR_MIN_OBJ_SIZE         32

#define     EO_MIN_FOV              0.006981f
#define     EO_MAX_FOV              0.392699f
#define     EO_MAX_SIZE             1920.0
#define     EO_STABLE_OBJ_SIZE      200.0



/** ===================================================
 *
 *              functions
 *
 * ====================================================
 */
extern cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2, int *size_patch, int feature_type, float sigma);

extern cv::Mat doubleGaussCorrelation(cv::Mat x1, cv::Mat x2, cv::Mat _alphaf, float sigma);

extern cv::Mat polynomialCorrelation(cv::Mat x1, cv::Mat x2, int *size_patch, int feature_type, float alpha, float beta);

extern cv::Mat createGaussianDistribution(int sizey, int sizex);

extern void createHannWindow(cv::Mat &_hann, int *size_patch, int featureType);

extern float fixPeak(float left , float center, float right);

extern float calcPsr(cv::Mat &response);

extern std::vector<int> limitRect(cv::Rect &window, cv::Size im_sz);

extern cv::Mat computeCorrelation(cv::Mat x1, cv::Mat x2);

#endif // UTILITIES_HPP

