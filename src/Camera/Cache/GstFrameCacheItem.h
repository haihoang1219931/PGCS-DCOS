#ifndef GSTFRAMECACHEITEM_H
#define GSTFRAMECACHEITEM_H

#include "CacheItem.h"
#include "gst/gst.h"
#include <memory>

namespace rva {
class GstFrameCacheItem : public CacheItem {
public:
  explicit GstFrameCacheItem() {}
  explicit GstFrameCacheItem(index_type _id) : CacheItem(_id) {}

    GstFrameCacheItem (const GstFrameCacheItem &_e){
        m_id = _e.m_id;
        m_gstFrame = _e.m_gstFrame;
    }

  ~GstFrameCacheItem() {}

//    std::shared_ptr<GstBuffer> getGstBuffer(){ return m_gstFrame; }
//    void setGstBuffer(std::shared_ptr<GstBuffer> _gstBuf){ m_gstFrame = _gstBuf; }

    GstBuffer* getGstBuffer(){ return m_gstFrame; }
    void setGstBuffer(GstBuffer *_gstBuf){ m_gstFrame = _gstBuf; }

  void release() {
      if(m_gstFrame != nullptr)
          gst_buffer_unref(m_gstFrame);
  }

  GstFrameCacheItem &operator=(const GstFrameCacheItem &_item) {
    this->m_id = _item.m_id;
    this->m_gstFrame = _item.m_gstFrame;
    return *this;
  }

private:
//  std::shared_ptr<GstBuffer> m_gstFrame;
  GstBuffer *m_gstFrame = nullptr;
};
}

#endif // GSTFRAMECACHEITEM_H
