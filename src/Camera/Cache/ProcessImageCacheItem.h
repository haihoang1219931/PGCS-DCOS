#ifndef PROCESSIMAGECACHEITEM_H
#define PROCESSIMAGECACHEITEM_H

#include "CacheItem.h"
#include <opencv2/opencv.hpp>
#include <QObject>
#include <vector>
#include <iostream>
#ifdef USE_VIDEO_GPU
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif

namespace rva
{
    class ProcessImageCacheItem : public CacheItem
    {
        public:
            explicit ProcessImageCacheItem() {}
            explicit ProcessImageCacheItem(index_type _id) : CacheItem(_id) {}

            ~ProcessImageCacheItem()
            {
            }

            void release()
            {

            }

            void setHostImage(unsigned char *_hImage)
            {
                m_hImage = _hImage;
            }

            unsigned char *getHostImage()
            {
                return m_hImage;
            }

            void setDeviceImage(unsigned char *_dImage)
            {
                m_dImage = _dImage;
            }

            unsigned char *getDeviceImage()
            {
                return m_dImage;
            }

            void setZoom(float zoom)
            {
                m_zoom = zoom;
            }

            float getZoom()
            {
                return m_zoom;
            }

            void setImageSize(cv::Size &_imgSize)
            {
                m_imgSize = _imgSize;
            }

            cv::Size getImageSize()
            {
                return m_imgSize;
            }

            void setHostStabMatrix(cv::Mat _hStabMatrix)
            {
                m_hStabMat[0] = static_cast<float>(_hStabMatrix.at<double>(0,0));
                m_hStabMat[1] = static_cast<float>(_hStabMatrix.at<double>(0,1));
                m_hStabMat[2] = static_cast<float>(_hStabMatrix.at<double>(0,2));
                m_hStabMat[3] = static_cast<float>(_hStabMatrix.at<double>(1,0));
                m_hStabMat[4] = static_cast<float>(_hStabMatrix.at<double>(1,1));
                m_hStabMat[5] = static_cast<float>(_hStabMatrix.at<double>(1,2));
                m_hStabMat[6] = static_cast<float>(_hStabMatrix.at<double>(2,0));
                m_hStabMat[7] = static_cast<float>(_hStabMatrix.at<double>(2,1));
                m_hStabMat[8] = static_cast<float>(_hStabMatrix.at<double>(2,2));
            }

            float* getHostStabMatrix()
            {
                return m_hStabMat;
            }

            void setDeviceStabMatrix(float *_dStabMatrix)
            {
                m_dStabMat = _dStabMatrix;
            }

            float *getDeviceStabMatrix()
            {
                return m_dStabMat;
            }

            void setHostGMEMatrix(float *_hGMEMatrix)
            {
                m_hGMEMat = _hGMEMatrix;
            }

            float *getHostGMEMatrix()
            {
                return m_hGMEMat;
            }

            void setDeviceGMEMatrix(float *_dGMEMatrix)
            {
                m_dGMEMat = _dGMEMatrix;
            }

            float *getDeviceGMEMatrix()
            {
                return m_dGMEMat;
            }

            int lockMode() {return m_lockMode;}
            void setLockMode(int lockMode){m_lockMode = lockMode;}
            cv::Rect trackRect() {return m_trackRect;}
            void setTrackRect(cv::Rect trackRect){m_trackRect = trackRect;}
            int trackStatus() {return m_trackStatus;}
            void setTrackStatus(int trackStatus){m_trackStatus = trackStatus;}
            cv::Rect powerlineDetectRect() {return m_powerlineDetectRect;}
            void setPowerlineDetectRect(cv::Rect powerlineDetectRect){m_powerlineDetectRect = powerlineDetectRect;}
            bool powerlineDetectEnable() {return m_powerlineDetectEnable;}
            void setPowerlineDetectEnable(bool powerlineDetectEnable){
                m_powerlineDetectEnable = powerlineDetectEnable;
            }
            std::vector<cv::Scalar> powerLineList(){return m_powerLineList;}
            void setPowerLineList(std::vector<cv::Scalar>& powerLineList){
                m_powerLineList = powerLineList;
            }
            int sensorID(){ return m_sensorID; }
            void setSensorID(int sensorID){ m_sensorID = sensorID; }

            int colorMode(){ return m_colorMode; }
            void setColorMode(int colorMode){ m_colorMode = colorMode; }
        private:
            unsigned char *m_dImage = nullptr;
            unsigned char *m_hImage = nullptr;
            cv::Size m_imgSize;
            float m_hStabMat[9];
            float m_zoom = 1.0f;
            float *m_dStabMat = nullptr;
            float *m_hGMEMat = nullptr;
            float *m_dGMEMat = nullptr;
            int m_lockMode = 0;
            int m_sensorID = 0;
            int m_colorMode = 0;
            cv::Rect m_trackRect;
            int m_trackStatus = 0;
            bool m_powerlineDetectEnable = false;
            cv::Rect m_powerlineDetectRect;
            std::vector<cv::Scalar> m_powerLineList;
    };
} // namespace rva

#endif // PROCESSIMAGECACHEITEM_H
