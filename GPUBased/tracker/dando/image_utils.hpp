#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP
#include<opencv2/opencv.hpp>
#include<math.h>
namespace ImageUtility
{
    cv::Mat contrastEnhance(cv::Mat image, float weightingParam = 0.5);
}
#endif // IMAGE_UTILS_HPP
