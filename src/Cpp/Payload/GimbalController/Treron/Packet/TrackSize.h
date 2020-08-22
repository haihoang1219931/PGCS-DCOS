#ifndef TrackSize_H
#define TrackSize_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
    class TrackSize :public Object
    {
    private:
        data_type m_size; //width region
    public:
        TrackSize()
        {
           m_size = 0;
        }

        TrackSize(const data_type _size)
        {
            m_size = _size;
        }

        TrackSize(const TrackSize &_TrackSize)
        {
            m_size = _TrackSize.m_size;
        }

        ~TrackSize(){}

        inline data_type getSize()const
        {
            return m_size;
        }

        inline void setSize(const data_type _size)
        {
            m_size = _size;
        }

        inline TrackSize & operator=(const TrackSize &_TrackSize)
        {
            m_size = _TrackSize.m_size;
            return *this;
        }

        inline bool operator==(const TrackSize &_TrackSize)
        {
            return (m_size == _TrackSize.m_size);
        }

        length_type size()
        {
            return 1 * sizeof(data_type);
        }

        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            std::vector<byte> b_size;

            b_size = Utils::toByte<data_type>(m_size);
            _result = b_size;
            return _result;
        }

        TrackSize* parse(std::vector<byte> _data, index_type _index = 0)
        {
            byte* data = _data.data();
            m_size = Utils::toValue<data_type>(data,_index);
            return this;
        }

        TrackSize* parse(byte* _data, index_type _index = 0)
        {
            m_size = Utils::toValue<data_type>(_data, 0);
            return this;
        }
    };
};

#endif // TrackSize_H
