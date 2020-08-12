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

            void setImageSize(cv::Size &_imgSize)
            {
                m_imgSize = _imgSize;
            }

            cv::Size getImageSize()
            {
                return m_imgSize;
            }

            void setHostStabMatrix(float *_hStabMatrix)
            {
                m_hStabMat = _hStabMatrix;
            }

            float *getHostStabMatrix()
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
            float *m_hStabMat = nullptr;
            float *m_dStabMat = nullptr;
            float *m_hGMEMat = nullptr;
            float *m_dGMEMat = nullptr;
    };
} // namespace rva

#endif // PROCESSIMAGECACHEITEM_H
