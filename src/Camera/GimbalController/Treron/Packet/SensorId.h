#ifndef SENSOR_ID_H
#define SENSOR_ID_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
    class SensorID : public Object
    {
        public:
            SensorID() {}
            SensorID(const byte _id)
            {
                m_id = _id;
            }

            ~SensorID() {}

            inline SensorID& operator=(const SensorID& _id)
            {
                m_id = _id.m_id;
                return *this;
            }

            inline bool operator==(const SensorID& _id)
            {
                return (m_id == _id.m_id);
            }

            inline byte getSensorId()const
            {
                return m_id;
            }

            inline void setSensorId(const byte _id)
            {
                m_id = _id;
            }
            length_type size()
            {
                return 1 * sizeof(byte);
            }
            std::vector<byte> toByte()
            {
                std::vector<byte> _result;
                _result.clear();
                _result.push_back(m_id);
                return _result;
            }
            SensorID* parse(std::vector<byte> _data, index_type _index = 0)
            {
                m_id = _data[_index];
                return this;
            }
            SensorID* parse(byte* _data, index_type _index = 0)
            {
                m_id = _data[_index];
                return this;
            }
        private:
            byte m_id;
    };
}
#endif
