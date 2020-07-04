#ifndef SENSOR_COLOR_H
#define SENSOR_COLOR_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
    class SensorColor :public Object
    {
    public:
        SensorColor(){}
        SensorColor(const byte _mode)
        {
            m_mode = _mode;
        }
        ~SensorColor(){}
    public:
        inline byte getMode() const{
            return m_mode;
        }
        inline void setMode(const byte _mode){
            m_mode = _mode;
        }
        inline SensorColor & operator=(const SensorColor &_mode)
        {
            m_mode = _mode.m_mode;
            return *this;
        }

        inline bool operator==(const SensorColor &_mode)
        {
            return (m_mode == _mode.m_mode);
        }

        length_type size()
        {
            return sizeof(m_mode);
        }

        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            _result.clear();
            _result.push_back(m_mode);
            return _result;
        }

        SensorColor* parse(std::vector<byte> _data, index_type _index = 0)
        {
            m_mode = _data[_index];
            return this;
        }

        SensorColor* parse(byte* _data, index_type _index = 0)
        {
            m_mode = _data[_index];
            return this;
        }

    private:
        byte m_mode;
    };
}
#endif
