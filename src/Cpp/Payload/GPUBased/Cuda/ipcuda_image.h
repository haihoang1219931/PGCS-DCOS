/**
  * @file ipcuda_image.h
  * @author Anh Dan <danda@viettel.com.vn/anhdan.do@gmail.com>
  * @version 1.0
  * @date 25 Jan 2019
  *
  * @section LICENSE
  *
  * @section DESCRIPTION
  *
  * This file contains the declarations of CUDA functions that perform
  * image color conversions between I420, Gray, RGB spaces; get image
  * patch from a rotated ROI on a larger image and histogram functions
  *
  * @see ipcuda_image.cu
  */

#ifndef IPCUDA_IMAGE_H
#define IPCUDA_IMAGE_H

#include <stdio.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

/**
 * @brief Converts I420 image to gray on GPU
 * @param d_i420    Device pointer to I420 image data
 * @param d_gray    Device pointer to gray image data
 * @param width     Image width
 * @param height    Image height
 * @return Completion status of the function
 */
extern cudaError_t gpu_i420ToGray( unsigned char *d_i420, unsigned char *d_gray, int width, int height );


/**
 * @brief Extract an image patch from I420 image concurrently with converting it to RGB on GPU
 * @param d_i420    Device pointer to I420 image data
 * @param d_rgb     Device pointer to RGB image patch data
 * @param width     Image width
 * @param height    Image height
 * @param roi_x     x-coordinate of the top-left corner of the patch
 * @param roi_y     y-coordinate of the top-left corner of the patch
 * @param roi_w     Patch width
 * @param roi_h     Patch height
 * @return Completion status of the function
 */
extern cudaError_t gpu_i420ToRGB( const unsigned char *d_i420, unsigned char *d_rgb, int width, int height,
                                  int roi_x, int roi_y, int roi_w, int roi_h );


/**
 * @brief Extract an image patch from I420 image concurrently with converting it to RGBA on GPU
 * @param d_i420    Device pointer to I420 image data
 * @param d_rgba    Device pointer to RGBA image patch data
 * @param width     Image width
 * @param height    Image height
 * @param roi_x     x-coordinate of the top-left corner of the patch
 * @param roi_y     y-coordinate of the top-left corner of the patch
 * @param roi_w     Patch width
 * @param roi_h     Patch height
 * @return Completion status of the function
 */
extern cudaError_t gpu_i420ToRGBA( const unsigned char *d_i420, unsigned char *d_rgba, int width, int height,
                                   int roi_x, int roi_y, int roi_w, int roi_h );


/**
 * @brief Converts an image of RGB format to I420 format
 * @param d_rgb     Device pointer to RGB image data
 * @param d_i420    Device pointer to I420 image data
 * @param width     Image width
 * @param height    Image height
 * @return
 */
extern cudaError_t gpu_rgbToI420( const unsigned char *d_rgb, unsigned char *d_i420, int width, int height );


/**
 * @brief RGBA to I420 image format conversion
 * @param d_rgba    Device pointer to RGBA image data
 * @param d_i420    Device pointer to I420 image data
 * @param width     Image width
 * @param height    Image height
 * @return Completion status of the function
 */
extern cudaError_t gpu_rgbaToI420( const unsigned char *d_rgba, unsigned char *d_i420, int width, int height );


/**
 * @brief UYVY to I420 image format conversion
 * @param d_uyvy    Device pointer to UYVY image data
 * @param d_i420    Device pointer to I420 image data
 * @param _width    Image width
 * @param _height   Image height
 * @return Completion status of the function
 */
extern cudaError_t gpu_uyvy2I420( const unsigned char *d_uyvy, unsigned char *d_i420, const int _width, const int _height );


/**
 * @brief YUYV to I420 image format conversion
 * @param d_yuyv    Device pointer to YUYV image data
 * @param d_i420    Device pointer to I420 image data
 * @param _width    Image width
 * @param _height   Image height
 * @return Completion status of the function
 */
extern cudaError_t gpu_yuyv2I420( const unsigned char *d_yuyv, unsigned char *d_i420, const int _width, const int _height );


/**
 * @brief Concurrently converts UYVY 4K image to I420 format and resize it to 2K resolution
 * @param d_uyvy4K  Device pointer to UYVY 4K image data
 * @param d_i4204K  Device pointer to I420 4K image data
 * @param d_i4202K  Device pointer to I420 2K image data
 * @return  Completion status of the function
 */
extern cudaError_t gpu_uyvy4K2I4204K2K( const unsigned char *d_uyvy4K, unsigned char *d_i4204K, unsigned char *d_i4202K );


/**
 * @brief Convert an unsigned char image to float one whose values are in range of [0.0; 1.0]
 * @param _ucharIm  Device pointer to unsigned char image
 * @param _floatIm  Device pointer to float image
 * @param _rows     Image rows
 * @param _cols     Image collumns
 * @return Completion status of the function
 */
extern cudaError_t gpu_im2float( unsigned char *_ucharIm, float *_floatIm, const int _rows, const int _cols );


/**
 * @brief Performs affine image warping on I420 format on GPU
 * @param d_src         Device pointer to input I420 image data
 * @param d_dst         Device pointer to output I420 image data
 * @param d_T           Device pointer to input transformation matrix data
 * @param _width        Image width
 * @param _height       Image height
 * @return Completion status of the function
 */
extern cudaError_t gpu_invWarpI420( const unsigned char *d_src, unsigned char *d_dst, const float *d_T, const int _width, const int _height );


/**
 * @brief Performs affine image warping on I420 format on GPU Version 2
 * @param d_src         Device pointer to input I420 image data
 * @param d_dst         Device pointer to output I420 image data
 * @param d_T           Device pointer to input transformation matrix data
 * @param _srcW         Source image width
 * @param _srcH         Source image height
 * @param _dstW         Destination image width
 * @param _dstH         Destination image height
 * @return Completion status of the function
 */
extern cudaError_t gpu_invWarpI420_V2( const unsigned char *d_src, unsigned char *d_dst, const float *d_T,
                                       const int _srcW, const int _srcH, const int _dstW, const int _dstH );


/**
 * @brief Perform image resizing on I420 format
 * @param d_i420Src     Device pointer to input I420 image data
 * @param d_i420Dst     Device pointer to output I420 image data
 * @param _srcW         Input image width
 * @param _srcH         Input image height
 * @param _dstW         Output image width
 * @param _dstH         Output image height
 * @return Completion status of the function
 */
extern cudaError_t gpu_i420Resize( const unsigned char *d_i420Src, unsigned char *d_i420Dst,
                                   const int _srcW, const int _srcH, const int _dstW, const int _dstH );


/**
 * @brief Extracts an image patch bounded by a rotated rectangle from a I420 image
 * @param _i420     Device pointer to gray image data
 * @param _patch    Device pointer to gray patch data
 * @param _iW       Image width
 * @param _iH       Image height
 * @param _rotRect  Rotated rectangle covering the patch
 * @param _scale    A dimension scale factor to resize the patch after the extraction
 * @return Completion status of the function
 */
extern cudaError_t gpu_getRotatedGrayPatchFromI420(unsigned char *_i420, unsigned char *_patch, const int _iW, const int _iH,
                                                   const cv::RotatedRect &_rotRect, const float _scale );


/**
 * @brief Concurrently extracts an image patch bounded by a rotated rectangle
 * from a I420 image and converts it to RGB format
 * @param _i420     Device pointer to input I420 image data
 * @param _rgbPatch Device pointer to input RGB image patch data
 * @param _iW       Image width
 * @param _iH       Image height
 * @param _rotRect  Rotated rectangle covering the patch
 * @param _scale    A dimension scale factor to resize the patch after the extraction
 * @return Completion status of the function
 */
extern cudaError_t gpu_getRotatedRGBPatchFromI420(unsigned char *_i420, unsigned char *_rgbPatch, const int _iW, const int _iH,
                                                  const cv::RotatedRect &_rotRect, const float _scale );

/**
 * @brief Copies a small RGB image patch to a specific location in a larger RGB image
 * @param _fromImg  Device pointer to small image patch
 * @param _toImg    Device pointer to larger image
 * @param _depth    Depth of images
 * @param _fx       Top-left corner x-coordinate
 * @param _fy       Top-left corner y-coordinate
 * @param _fWidth   Patch width
 * @param _fHeight  Patch height
 * @param _tWidth   Image width
 * @param _tHeight  Image height
 * @return Completion status of the function
 */
extern cudaError_t gpu_copyRoi( unsigned char *_fromImg, unsigned char *_toImg, const int _depth,
                                const int _fx, int _fy, int _fWidth, int _fHeight, int _tWidth, int _tHeight );


/**
 * @brief  Sets values of pixels in a specific region of an I420 image to values of another I420 image
 * whose size is equal to the size of the above region
 * @param d_i420Src     Device pointer to source I420 image
 * @param d_i420Dst     Device pointer to destination I420 image
 * @param _x            Top-left x coordinate of image ROI
 * @param _y            Top-left y coordinate of image ROI
 * @param _srcWidth     Source image width
 * @param _srcHeight    Source image height
 * @param _dstWidth     Destination image width
 * @param _dstHeight    Destination image height
 * @return Completion status of the function
 */
extern cudaError_t gpu_setI420ROIData( const unsigned char *d_i420Src, unsigned char *d_i420Dst, const int _x, const int _y,
                                       const int _srcWidth, const int _srcHeight, const int _dstWidth, const int _dstHeight );


/**
 * @brief Computes normalized histgram of an image.
 *
 * Before the normalization step, if the image is multi-channels, histograms are computed
 * separately for each channel and then concanated to form the image histogram array.
 * A Hanning window is applied to weight values of pixels closer to the image center
 * higher than those closer to the edges
 *
 * @param _img      Device pointer to input image data
 * @param _hann     Device pointer to Hanning window data
 * @param _hist     Device pointer to histogram array
 * @param _width    Image width
 * @param _height   Image height
 * @param _depth    Image depth (in channels)
 * @param _binNum   Number of bins for the histogram of ech channel
 * @return  Completion status of the function
 */
extern cudaError_t gpu_histogram( const unsigned char *_img, const float *_hann, float *_hist,
                                  const int _width, const int _height, const int _depth, const int _binNum );


/**
 * @brief Computes histogram similarity score between two histogram arrays
 *
 * The score is computed as a Bhattacharyya coefficienct and is measured
 * in degree (0 <= _score <= 90)
 *
 * @param _hist1    Device pointer to the first histogram array data
 * @param _hist2    Device pointer to the second histogram array data
 * @param _score    Host pointer to similarity score variable
 * @param _len      Lengths of the arrays
 * @return Completion status of the function
 */
extern cudaError_t gpu_histSimilarity( const float *_hist1, const float *_hist2, float *_score, const int _len );

#endif // IPCUDA_IMAGE_H

