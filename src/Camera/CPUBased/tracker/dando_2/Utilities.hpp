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

#define     SCALE_CHANGE_RATIO      1.25f
#define     SCALE_WEIGTH            0.95f

#define     OCCLUDED_THRESH_GAUSS   0.6f
#define     OCCLUDED_THRESH_POLY    600

#define     CELL_SIZE               4
#define     TEMPLATE_SIZE           96

#define     KERNEL_GAUSSIAN         0
#define     KERNEL_POLYNOMIAL       1

#define     FEATURE_HOG             0
#define     FEATURE_GRAY            1


#define     FOUND                   0
#define     LOST                    1



/** ===================================================
 *
 *              functions
 *
 * ====================================================
 */
extern cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2, int *size_patch, int feature_type, float sigma);

extern cv::Mat doubleGaussCorrelation(cv::Mat x1, cv::Mat x2, cv::Mat _alphaf, float sigma );

extern cv::Mat polynomialCorrelation(cv::Mat x1, cv::Mat x2, int *size_patch, int feature_type, float alpha, float beta );

extern cv::Mat createGaussianDistribution(int sizey, int sizex);

extern void createHannWindow( cv::Mat &_hann, int *size_patch, int featureType );

extern float fixPeak(float left , float center, float right);

extern float calcPsr(cv::Mat &response);

extern std::vector<int> limitRect(cv::Rect &window, cv::Size im_sz);

#endif // UTILITIES_HPP

