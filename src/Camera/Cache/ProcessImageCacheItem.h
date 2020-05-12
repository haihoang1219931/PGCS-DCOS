#ifndef PROCESSIMAGECACHEITEM_H
#define PROCESSIMAGECACHEITEM_H

#include "CacheItem.h"
#include <opencv2/opencv.hpp>
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
                m_hStabMat[0] = _hStabMatrix.at<float>(0,0);
                m_hStabMat[1] = _hStabMatrix.at<float>(0,1);
                m_hStabMat[2] = _hStabMatrix.at<float>(0,2);
                m_hStabMat[3] = _hStabMatrix.at<float>(1,0);
                m_hStabMat[4] = _hStabMatrix.at<float>(1,1);
                m_hStabMat[5] = _hStabMatrix.at<float>(1,2);
                m_hStabMat[6] = _hStabMatrix.at<float>(2,0);
                m_hStabMat[7] = _hStabMatrix.at<float>(2,1);
                m_hStabMat[8] = _hStabMatrix.at<float>(2,2);
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



        private:
            unsigned char *m_dImage = nullptr;
            unsigned char *m_hImage = nullptr;
            cv::Size m_imgSize;
            float m_hStabMat[9];
            float m_zoom = 1.0f;
            float *m_dStabMat = nullptr;
            float *m_hGMEMat = nullptr;
            float *m_dGMEMat = nullptr;
            cv::Rect m_trackRect;
    };
} // namespace rva

#endif // PROCESSIMAGECACHEITEM_H
