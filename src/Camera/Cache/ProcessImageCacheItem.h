#ifndef PROCESSIMAGECACHEITEM_H
#define PROCESSIMAGECACHEITEM_H

#include "CacheItem.h"
#include <opencv2/opencv.hpp>
#ifdef GPU_PROCESS
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



        private:
            unsigned char *m_dImage = nullptr;
            unsigned char *m_hImage = nullptr;
            cv::Size m_imgSize;
    };
} // namespace rva

#endif // PROCESSIMAGECACHEITEM_H
