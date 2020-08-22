#ifndef CACHEITEM_H
#define CACHEITEM_H

#include "Payload/Packet/Common_type.h"
using namespace Eye;
namespace rva {
class CacheItem {
public:
  explicit CacheItem() {}
  explicit CacheItem(index_type _id) : m_id(_id) {}
  ~CacheItem(){}

  void setIndex(const index_type &_id) { m_id = _id; }
  index_type getIndex() { return m_id; }

  void release() {}

  CacheItem &operator=(const CacheItem &_item) {
    this->m_id = _item.m_id;
    return *this;
  }

protected:
  index_type m_id = 0;
};
}

#endif // CACHEITEM_H
