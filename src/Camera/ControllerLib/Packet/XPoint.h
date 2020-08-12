#ifndef X_POINT_H
#define X_POINT_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye {
class XPoint : public Object {
private:
  index_type m_id;
  data_type m_px;     // x - pixel of screen
  data_type m_py;     // y - pixel of screen
  data_type m_width;  // width region
  data_type m_height; // height region
public:
  XPoint() {
    m_id = -1;
    m_px = 0;
    m_py = 0;
    m_width = 0;
    m_height = 0;
  }

  XPoint(const index_type _id, const data_type _px, const data_type _py,
         const data_type _width = 0, const data_type _height = 0) {
    m_id = _id;
    m_px = _px;
    m_py = _py;
    m_width = _width;
    m_height = _height;
  }

  XPoint(const XPoint &_point) {
    m_id = _point.m_id;
    m_px = _point.m_px;
    m_py = _point.m_py;
    m_width = _point.m_width;
    m_height = _point.m_height;
  }

  ~XPoint() {}
  inline index_type getIndex() const { return m_id; }

  inline data_type getPx() const { return m_px; }
  inline data_type getPy() const { return m_py; }

  inline data_type getWidth() const { return m_width; }

  inline data_type getHeight() const { return m_height; }

  inline void setId(const index_type _id) { m_id = _id; }
  inline void setLocation(const data_type _px, const data_type _py) {
    m_px = _px;
    m_py = _py;
  }

  inline void setRegion(const data_type _width, const data_type _height) {
    m_width = _width;
    m_height = _height;
  }

  inline void setXPoint(const index_type _id, const data_type _px,
                        const data_type _py, const data_type _width,
                        const data_type _height) {
    m_id = _id;
    m_px = _px;
    m_py = _py;
    m_width = _width;
    m_height = _height;
  }

  inline XPoint &operator=(const XPoint &_point) {
    m_id = _point.m_id;
    m_px = _point.m_px;
    m_py = _point.m_py;
    m_width = _point.m_width;
    m_height = _point.m_height;
    return *this;
  }

  inline bool operator==(const XPoint &_point) {
    return (m_id == _point.m_id && m_px == _point.m_px && m_py == _point.m_py &&
            m_width == _point.m_width && m_height == _point.m_height);
  }

  length_type size() { return 4 * sizeof(data_type) + sizeof(index_type); }

  std::vector<byte> toByte() {
    std::vector<byte> _result;
    std::vector<byte> b_id, b_px, b_py, b_width, b_height;

    b_id = Utils::toByte<index_type>(m_id);
    b_px = Utils::toByte<data_type>(m_px);
    b_py = Utils::toByte<data_type>(m_py);
    b_width = Utils::toByte<data_type>(m_width);
    b_height = Utils::toByte<data_type>(m_height);

    _result = b_id;
    _result.insert(_result.end(), b_px.begin(), b_px.end());
    _result.insert(_result.end(), b_py.begin(), b_py.end());
    _result.insert(_result.end(), b_width.begin(), b_width.end());
    _result.insert(_result.end(), b_height.begin(), b_height.end());
    return _result;
  }

  XPoint *parse(std::vector<byte> _data, index_type _index = 0) {
    byte *data = _data.data();
    m_id = Utils::toValue<index_type>(data, _index);
    m_px = Utils::toValue<data_type>(data, _index + sizeof(index_type));
    m_py = Utils::toValue<data_type>(data, _index + sizeof(index_type) +
                                               sizeof(data_type));
    m_width = Utils::toValue<data_type>(data, _index + sizeof(index_type) +
                                                  2 * sizeof(data_type));
    m_height = Utils::toValue<data_type>(data, _index + sizeof(index_type) +
                                                   3 * sizeof(data_type));
    return this;
  }

  XPoint *parse(byte *_data, index_type _index = 0) {
    m_id = Utils::toValue<index_type>(_data, _index);
    m_px = Utils::toValue<data_type>(_data, _index + sizeof(index_type));
    m_py = Utils::toValue<data_type>(_data, _index + sizeof(index_type) +
                                                sizeof(data_type));
    m_width = Utils::toValue<data_type>(_data, _index + sizeof(index_type) +
                                                   2 * sizeof(data_type));
    m_height = Utils::toValue<data_type>(_data, _index + sizeof(index_type) +
                                                    3 * sizeof(data_type));
    return this;
  }
};
}; // namespace Eye
#endif
