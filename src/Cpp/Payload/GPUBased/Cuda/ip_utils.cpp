/**
  * @file ip_commonUtils.cpp
  * @author Anh Dan <danda@viettel.com.vn/anhdan.do@gmail.com>
  * @version 1.0
  * @date 24 Jan 2019
  *
  * @section LICENSE
  *
  * @section DESCRIPTION
  *
  * This file contains the definition of commonly used functions and constants,
  * including matrix/array operation functions
  *
  * @see ip_commonUtils.h
  */

#include "ip_utils.h"

namespace ip
{
    /**
     * @brief composeTransform
     */
    cv::Mat composeTransform(const float _dx, const float _dy, const float _rot, const float _scale)
    {
        cv::Mat trans = cv::Mat::eye(3, 3, CV_32FC1);
        trans.at<float>(2, 0) = _dx;
        trans.at<float>(2, 1) = _dy;
        trans.at<float>(0, 0) = trans.at<float>(1, 1) = _scale * cosf(_rot);
        trans.at<float>(1, 0) = _scale * sinf(_rot);
        trans.at<float>(0, 1) = -trans.at<float>(1, 0);
        return trans;
    }

    /**
     * @brief isImageTextureless
     */
    bool isImageTextureless(const unsigned char *d_gray, const int _width, const int _height)
    {
        if (d_gray == NULL) {
            return true;
        }

        bool status;
        cv::Scalar meanVal, stdVal;
#if CV_MAJOR_VERSION < 3
        cv::gpu::GpuMat gpuImg(_height, _width, CV_8UC1, (void *)d_gray);
        cv::gpu::meanStdDev(gpuImg, meanVal, stdVal);
#else
        cv::cuda::GpuMat gpuImg(_height, _width, CV_8UC1, (void *)d_gray);
        cv::cuda::meanStdDev(gpuImg, meanVal, stdVal);
#endif

        if (meanVal[0] == 0) {
            status = true;
        } else {
            double variance = stdVal[0] / meanVal[0];

            if (variance <= PATCH_MIN_VARIANCE) {
                status = true;
            } else {
                status = false;
            }
        }

        gpuImg.release();
        return status;
    }


    //! FIXIT: All the bellow functions are for debugging only, and should be deleted when the project is completed

    /**
     * @brief writeMatToText
     */
    void writeMatToText(const char *_filename , float *_ptr, const int _rows, const int _cols, bool _isGPUMat)
    {
        cv::Mat hMat(_rows, _cols, CV_32FC1);

        if (_isGPUMat) {
            cudaMemcpy(hMat.data, _ptr, _rows * _cols * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            memcpy(hMat.data, _ptr, _rows * _cols * sizeof(float));
        }

        usleep(100);
        std::ofstream fp(_filename);

        if (!fp) {
            LOG_MSG("! ERROR: %s:%d: Failed to open file\n");
            return;
        }

        for (int i = 0; i < _rows; i++) {
            for (int j = 0; j < _cols; j++) {
                fp << hMat.at<float>(i, j) << ' ';
            }

            fp << std::endl;
        }

        fp.close();
    }


    /**
     * @brief dispGPUImage
     */
    void dispGPUImage(const void *_imgPtr, const int _width, const int _height, const int _type, const std::string &_winName)
    {
#if CV_MAJOR_VERSION < 3
        cv::gpu::GpuMat gTmp(_height, _width, _type, (void *)_imgPtr);
#else
        cv::cuda::GpuMat gTmp(_height, _width, _type, (void *)_imgPtr);
#endif
        cv::Mat cTmp;
        gTmp.download(cTmp);

        if (_type == CV_32FC1) {
            double minVal, maxVal;
            cv::minMaxLoc(cTmp, &minVal, &maxVal, NULL, NULL);

            if (minVal != maxVal)
                cTmp = (cTmp - minVal) / (maxVal - minVal);
        }

        cv::imshow(_winName, cTmp);
    }
}
