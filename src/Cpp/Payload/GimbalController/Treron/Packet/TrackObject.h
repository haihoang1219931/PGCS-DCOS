#ifndef TRACKOBJECT_H
#define TRACKOBJECT_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
    class TrackObject :public Object
    {
    private:
        index_type m_id;
        data_type m_px; // x - pixel of screen
        data_type m_py; // y - pixel of screen
        data_type m_width; //width region
        data_type m_height; // height region
        data_type m_objWidth; // object track width
        data_type m_objHeight; // object track height

    public:
        TrackObject()
        {
            m_id = -1;
            m_px = 0;
            m_py = 0;
            m_width = 0;
            m_height = 0;
            m_objWidth = 0;
            m_objHeight = 0;
        }

        TrackObject(const index_type _id, const data_type _px, const data_type _py,
                      const data_type _width = 0, const data_type _height = 0,
                      const data_type _objWidth = 0, const data_type _objHeight = 0)
        {
            m_id = _id;
            m_px = _px;
            m_py = _py;
            m_width = _width;
            m_height = _height;
            m_objHeight = _objHeight;
            m_objWidth = _objWidth;
        }

        TrackObject(const TrackObject &_TrackObject)
        {
            m_id = _TrackObject.m_id;
            m_px = _TrackObject.m_px;
            m_py = _TrackObject.m_py;
            m_width = _TrackObject.m_width;
            m_height = _TrackObject.m_height;
            m_objWidth = _TrackObject.m_objWidth;
            m_objHeight = _TrackObject.m_objHeight;
        }

        ~TrackObject(){}

        inline index_type getIndex()const
        {
            return m_id;
        }

        inline data_type getPx()const
        {
            return m_px;
        }
        inline data_type getPy()const
        {
            return m_py;
        }

        inline data_type getWidth()const
        {
            return m_width;
        }

        inline data_type getHeight()const
        {
            return m_height;
        }

        inline data_type getObjWidth()const
        {
            return m_objWidth;
        }

        inline data_type getObjHeight()const
        {
            return m_objHeight;
        }

        inline void setId(const index_type _id)
        {
            m_id = _id;
        }
        inline void setLocation(const data_type _px, const data_type _py)
        {
            m_px = _px;
            m_py = _py;
        }

        inline void setRegion(const data_type _width, const data_type _height)
        {
            m_width = _width;
            m_height = _height;
        }

        inline void setTrackObject(const index_type _id,const data_type _px, const data_type _py,
                                     const data_type _width, const data_type _height,
                                     const data_type _objWidth, const data_type _objHeight)
        {
            m_id = _id;
            m_px = _px;
            m_py = _py;
            m_width = _width;
            m_height = _height;
            m_objWidth = _objWidth;
            m_objHeight = _objHeight;
        }

        inline TrackObject & operator=(const TrackObject &_TrackObject)
        {
            m_id = _TrackObject.m_id;
            m_px = _TrackObject.m_px;
            m_py = _TrackObject.m_py;
            m_width = _TrackObject.m_width;
            m_height = _TrackObject.m_height;
            m_objWidth = _TrackObject.m_objWidth;
            m_objHeight = _TrackObject.m_objHeight;
            return *this;
        }

        inline bool operator==(const TrackObject &_TrackObject)
        {
            return (m_id == _TrackObject.m_id&&m_px == _TrackObject.m_px&&
                    m_py == _TrackObject.m_py&&m_width == _TrackObject.m_width&&
                    m_height == _TrackObject.m_height&&
                    m_objHeight == _TrackObject.m_objHeight && m_objWidth == _TrackObject.m_objWidth);
        }

        length_type size()
        {
            return 6 * sizeof(data_type)+sizeof(index_type);
        }

        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            std::vector<byte> b_id,b_px, b_py, b_width,b_height, b_objWidth, b_objHeight;

            b_id = Utils::toByte<index_type>(m_id);
            b_px = Utils::toByte<data_type>(m_px);
            b_py = Utils::toByte<data_type>(m_py);
            b_width = Utils::toByte<data_type>(m_width);
            b_height = Utils::toByte<data_type>(m_height);
            b_objWidth = Utils::toByte<data_type>(m_objWidth);
            b_objHeight = Utils::toByte<data_type>(m_objHeight);

            _result = b_id;
            _result.insert(_result.end(), b_px.begin(), b_px.end());
            _result.insert(_result.end(), b_py.begin(), b_py.end());
            _result.insert(_result.end(), b_width.begin(), b_width.end());
            _result.insert(_result.end(), b_height.begin(), b_height.end());
            _result.insert(_result.end(), b_objWidth.begin(), b_objWidth.end());
            _result.insert(_result.end(), b_objHeight.begin(), b_objHeight.end());

            return _result;
        }

        TrackObject* parse(std::vector<byte> _data, index_type _index = 0)
        {
            byte* data = _data.data();
            m_id = Utils::toValue<index_type>(data,_index);
            m_px = Utils::toValue<data_type>(data, _index + sizeof(index_type));
            m_py = Utils::toValue<data_type>(data, _index + sizeof(index_type)+sizeof(data_type));
            m_width = Utils::toValue<data_type>(data, _index + sizeof(index_type)+2*sizeof(data_type));
            m_height = Utils::toValue<data_type>(data, _index + sizeof(index_type)+3*sizeof(data_type));
            m_objWidth = Utils::toValue<data_type>(data, _index + sizeof(index_type)+4*sizeof(data_type));
            m_objHeight = Utils::toValue<data_type>(data, _index + sizeof(index_type)+5*sizeof(data_type));
            return this;
        }

        TrackObject* parse(byte* _data, index_type _index = 0)
        {
            m_id = Utils::toValue<index_type>(_data, _index);
            m_px = Utils::toValue<data_type>(_data, _index + sizeof(index_type));
            m_py = Utils::toValue<data_type>(_data, _index + sizeof(index_type)+sizeof(data_type));
            m_width = Utils::toValue<data_type>(_data, _index + sizeof(index_type)+2*sizeof(data_type));
            m_height = Utils::toValue<data_type>(_data, _index + sizeof(index_type)+3*sizeof(data_type));
            m_objWidth = Utils::toValue<data_type>(_data, _index + sizeof(index_type)+4*sizeof(data_type));
            m_objHeight = Utils::toValue<data_type>(_data, _index + sizeof(index_type)+5*sizeof(data_type));
            return this;
        }
    };
};

#endif // TRACKOBJECT_H
