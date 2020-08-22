/**
  * @file ip_commonUtils.h
  * @author Anh Dan <danda@viettel.com.vn/anhdan.do@gmail.com>
  * @version 1.0
  * @date 24 Jan 2019
  *
  * @section LICENSE
  *
  * @section DESCRIPTION
  *
  * This file contains the declaration of commonly used functions and constants,
  * including matrix/array operation functions
  *
  * @see ip_commonUtils.cpp
  */

#ifndef IP_COMMONUTILS_H
#define IP_COMMONUTILS_H

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#if CV_MAJOR_VERSION < 3
#include <opencv2/gpu/gpu.hpp>
#endif

//! FIXIT: The bellow libs are only for debugging, deleted them after completing the project
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

#define DEBUG

#ifdef DEBUG
#define LOG_MSG( ... )      printf( __VA_ARGS__ )
#else
#define LOG_MSG( ... )
#endif

#define NUMARGS(...)  (sizeof((int[]){0, ##__VA_ARGS__})/sizeof(int)-1)

#define isInside(x, y, w, h) (((x) >= 0) && ((x) < w) && ((y) >= 0) && ((y) < h))

#define CLEAR(x) memset(&(x), 0, sizeof(x))

#ifndef THREAD_PER_BLOCK
#define THREAD_PER_BLOCK    512
#endif

#ifndef ZERO_EPSILON
#define ZERO_EPSILON        1e-6
#endif

#define PATCH_MIN_VARIANCE  0.10f

// For VideoReadingThread
#define BUFFER_LENGTH       10    /**< Number of buffers to store temporary image data comming in continuously */

/**
 * @brief The camera IO method
 */
enum IOMethod {
    IO_METHOD_READ,
    IO_METHOD_MMAP,
    IO_METHOD_USERPTR,
};

/**
 * @brief The Buffer struct
 */
struct Buffer {
    void   *start;
    size_t  length;
};

namespace ip
{
    /**
    * @brief Composes a transformation matrix
    *
    * The transform matrix is similarity type and formated as following
    *      T = [ a1  a2   0
    *
    *           -a2  a1   0
    *
    *            dx  dy   1];
    *
    * @param _dx       x-transition component
    * @param _dy       y-transition component
    * @param _rot      Rotation component
    * @param _scale    Scale component
    * @return OpenCV transformation matrix
    */
    extern cv::Mat composeTransform(const float _dx, const float _dy, const float _rot, const float _scale);


    /**
     * @brief Checks if a gray scale image is textureless
     * @param d_gray    Device pointer to gray image data
     * @param _width    Image width
     * @param _height   Image height
     * @return Boolean value indicating image is textureless or not
     */
    extern bool isImageTextureless(const unsigned char *d_gray, const int _width, const int _height);


    //! FIXIT: All the bellow functions are for debugging only, and should be deleted when the project is completed
    extern void writeMatToText(const char *_filename , float *_ptr, const int _rows, const int _cols, bool _isGPUMat);

    extern void dispGPUImage(const void *_imgPtr, const int _width, const int _height, const int _type, const std::string &_winName);
}

#endif // IP_COMMONUTILS_H

