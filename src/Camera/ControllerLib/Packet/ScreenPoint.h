#ifndef SCREEN_POINT_H
#define SCREEN_POINT_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"
#include "RTData.h"


namespace Eye {
    class ScreenPoint :public Object
    {
    private:
        index_type m_id;
        data_type m_x; // x - x axis in coordinates according to GCS screen
        data_type m_y; // y - y axis in pixel of screen
        data_type m_width; //width region
        data_type m_height;// height region

    public:
        ScreenPoint(){
            m_id = -1;
            m_x = 0;
            m_y = 0;
            m_width = 0;
            m_height = 0;
        }

        ScreenPoint(const index_type _id, const data_type _x, const data_type _y, const data_type _width = 0, const data_type _height = 0)
        {
            m_id = _id;
            m_x = _x;
            m_y = _y;
            m_width = _width;
            m_height = _height;
        }

        ScreenPoint(const ScreenPoint &_point)
        {
            m_id = _point.m_id;
            m_x = _point.m_x;
            m_y = _point.m_y;
            m_width = _point.m_width;
            m_height = _point.m_height;
        }

        ~ScreenPoint(){}

        inline index_type getId()const
        {
            return m_id;
        }

        inline data_type getX()const
        {
            return m_x;
        }
        inline data_type getY()const
        {
            return m_y;
        }

        inline data_type getWidth()const
        {
            return m_width;
        }

        inline data_type getHeight()const
        {
            return m_height;
        }

        inline void setId(const index_type _id)
        {
            m_id = _id;
        }
        inline void setLocation(const data_type _x, const data_type _y)
        {
            m_x = _x;
            m_y = _y;
        }

        inline void setRegion(const data_type _width, const data_type _height)
        {
            m_width = _width;
            m_height = _height;
        }

        inline ScreenPoint & operator=(const ScreenPoint &_point)
        {
            m_id = _point.m_id;
            m_x = _point.m_x;
            m_y = _point.m_y;
            m_width = _point.m_width;
            m_height = _point.m_height;
            return *this;
        }

        inline bool operator==(const ScreenPoint &_point)
        {
            return (m_id == _point.m_id&&
                m_x == _point.m_x&&m_y == _point.m_y&&m_width==_point.m_width&&m_height==_point.m_height);
        }

        length_type size()
        {
            return 4 * sizeof(data_type)+sizeof(index_type);
        }

        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            std::vector<byte> b_id, b_x, b_y, b_width,b_height;

            b_id = Utils::toByte<index_type>(m_id);
            b_x = Utils::toByte<data_type>(m_x);
            b_y = Utils::toByte<data_type>(m_y);
            b_width = Utils::toByte<data_type>(m_width);
            b_height = Utils::toByte<data_type>(m_height);

            _result = b_id;
            _result.insert(_result.end(), b_x.begin(), b_x.end());
            _result.insert(_result.end(), b_y.begin(), b_y.end());
            _result.insert(_result.end(), b_width.begin(), b_width.end());
            _result.insert(_result.end(), b_height.begin(), b_height.end());
            return _result;
        }

        ScreenPoint* parse(std::vector<byte> _data, index_type _index = 0)
        {
            byte* data = _data.data();
            m_id = Utils::toValue<index_type>(data,_index);
            m_x = Utils::toValue<data_type>(data, _index + sizeof(index_type));
            m_y = Utils::toValue<data_type>(data, _index + sizeof(index_type)+sizeof(data_type));
            m_width = Utils::toValue<data_type>(data, _index + sizeof(index_type)+2*sizeof(data_type));
            m_height = Utils::toValue<data_type>(data, _index + sizeof(index_type)+3*sizeof(data_type));
            return this;
        }

        ScreenPoint* parse(byte* _data, index_type _index = 0)
        {
            m_id = Utils::toValue<index_type>(_data, _index);
            m_x = Utils::toValue<data_type>(_data, _index + sizeof(index_type));
            m_y = Utils::toValue<data_type>(_data, _index + sizeof(index_type)+sizeof(data_type));
            m_width = Utils::toValue<data_type>(_data, _index + sizeof(index_type)+2 * sizeof(data_type));
            m_height = Utils::toValue<data_type>(_data, _index + sizeof(index_type)+3 * sizeof(data_type));
            return this;
        }
    };
}
#endif
