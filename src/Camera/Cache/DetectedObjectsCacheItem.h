#ifndef DETECTEDOBJECTSCACHEITEM_H
#define DETECTEDOBJECTSCACHEITEM_H

#ifdef USE_VIDEO_GPU
    #include "../GPUBased/OD/yolo_v2_class.hpp"
#endif
#include "CacheItem.h"
#include "gst/gst.h"
#include <memory>

namespace rva
{
    class DetectedObjectsCacheItem : public CacheItem
    {
        public:
            explicit DetectedObjectsCacheItem() {}
            explicit DetectedObjectsCacheItem(index_type _id) : CacheItem(_id) {}

            DetectedObjectsCacheItem(const DetectedObjectsCacheItem &_e)
            {
                m_id = _e.m_id;
                m_listObj = _e.m_listObj;
            }

            ~DetectedObjectsCacheItem() {}

            std::vector<bbox_t> getDetectedObjects()
            {
                return m_listObj;
            }
            void setDetectedObjects(std::vector<bbox_t> _listObj)
            {
                m_listObj = _listObj;
            }

            void release() {}

            DetectedObjectsCacheItem &operator=(const DetectedObjectsCacheItem &_item)
            {
                this->m_id = _item.m_id;
                this->m_listObj = _item.m_listObj;
                return *this;
            }

        private:
            std::vector<bbox_t> m_listObj;
    };
} // namespace rva

#endif // DETECTEDOBJECTSCACHEITEM_H
