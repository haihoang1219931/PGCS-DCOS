#ifndef MOTIONIMAGE_H
#define MOTIONIMAGE_H

#include "Common_type.h"
#include "MData.h"

namespace Eye {
class MotionImage {
private:
  index_type m_index;
  MData m_motionContext;
  MData m_motionStab;

public:
  MotionImage() { m_index = -1; }

  MotionImage(const index_type _index, const MData &_motionContext,
              const MData &_motionStab) {
    m_index = _index;
    m_motionContext = _motionContext;
    m_motionStab = _motionStab;
  }

  ~MotionImage() {}

  void setMotionContext(const MData &_motionContext) {
    m_motionContext = _motionContext;
  }

  void setMotionStab(const MData &_motionStab) { m_motionStab = _motionStab; }

  void setIndex(const index_type &_index) { m_index = _index; }

  index_type getIndex() { return m_index; }

  MData getMotionContext() { return m_motionContext; }

  MData getMotionStab() { return m_motionStab; }

  std::vector<byte> toByte() {
    std::vector<byte> result;
    std::vector<byte> b_index, b_motionContext, b_motionStab;
    b_index = Utils::toByte<index_type>(m_index);
    b_motionContext = m_motionContext.toByte();
    b_motionStab = m_motionStab.toByte();

    result = b_index;
    result.insert(result.end(), b_motionContext.begin(), b_motionContext.end());
    result.insert(result.end(), b_motionStab.begin(), b_motionStab.end());
    return result;
  }

  MotionImage *parse(std::vector<byte> _data, index_type _index = 0) {
    byte *data = _data.data();
    if (_data.size() < _index + sizeof(m_index)) {
      return this;
    }
    m_index = Utils::toValue<index_type>(data, _index);
    m_motionContext.parse(data, _index + sizeof(index_type));
    m_motionStab.parse(data,
                       _index + sizeof(index_type) + m_motionContext.size());
    return this;
  }

  MotionImage *parse(byte *_data, index_type _index = 0) {
    m_index = Utils::toValue<index_type>(_data, _index);
    m_motionContext.parse(_data, _index + sizeof(index_type));
    m_motionStab.parse(_data,
                       _index + sizeof(index_type) + m_motionContext.size());
    return this;
  }

  MotionImage &operator=(const MotionImage &_motionImage) {
    m_index = _motionImage.m_index;
    m_motionContext = _motionImage.m_motionContext;
    m_motionStab = _motionImage.m_motionStab;
    return *this;
  }
};
} // namespace Eye

#endif // MOTIONIMAGE_H
