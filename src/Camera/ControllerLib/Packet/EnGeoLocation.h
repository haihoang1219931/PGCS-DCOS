#ifndef ENGEOLOCATION_H
#define ENGEOLOCATION_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"


namespace Eye
{
    class EnGeoLocation:public Object
    {
    public:
        EnGeoLocation(){
            m_en = true;
        }
        EnGeoLocation(bool _en) : m_en(_en)
        {
        }
        ~EnGeoLocation()
        {
        }

        inline EnGeoLocation & operator=(const EnGeoLocation &_en)
        {
            m_en = _en.m_en;
            return *this;
        }

        inline bool operator==(const EnGeoLocation &_en)
        {
            return (m_en == _en.m_en);
        }


        inline void setEnGeoLocation(const bool _en)
        {
            m_en = _en;
        }

        inline bool getEnGeoLocation(){
            return m_en;
        }

        length_type size()
        {
            return sizeof(m_en);
        }
        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            _result.clear();
            _result.push_back((byte)m_en);
            return _result;
        }
        EnGeoLocation* parse(std::vector<byte> _data, index_type _index = 0)
        {
            m_en = (bool) _data.at(_index);
            return this;
        }
        EnGeoLocation* parse(byte* _data, index_type _index = 0)
        {
            m_en = (bool) _data[_index];
            return this;
        }
    private:
        bool m_en;
    };
};

#endif // ENGEOLOCATION_H
