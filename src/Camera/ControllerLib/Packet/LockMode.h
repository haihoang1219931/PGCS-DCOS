#ifndef LOCK_MODE_H
#define LOCK_MODE_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"
#include "EyeStatus.h"

namespace Eye
{
    class LockMode:public Object
    {
    private:
        byte m_lock; // Virtual lock, Track, or GeoLock
        byte m_location;
    public:
        LockMode():m_lock((byte)Status::LockMode::LOCK_OFF), m_location((byte)Status::GeolocationMode::GEOLOCATION_OFF){}
        LockMode(byte _lock, byte _location = (byte) Status::GeolocationMode::GEOLOCATION_OFF)
        {
            m_lock = _lock;
            m_location = _location;
        }
        LockMode(LockMode &_mode)
        {
            m_lock = _mode.m_lock;
            m_location = _mode.m_location;
        }
        ~LockMode(){}

        inline LockMode & operator=(const LockMode &_mode)
        {
            m_lock = _mode.m_lock;
            m_location = _mode.m_location;
            return *this;
        }

        inline bool operator==(const LockMode &_mode)
        {
            return (m_lock == _mode.m_lock&&m_location==_mode.m_location);
        }

        inline byte getLockMode()const
        {
            return m_lock;
        }

        inline byte getGeoLocationMode()const
        {
            return m_location;
        }

        inline void setLockMode(const byte _lock, const byte _location = (byte) Status::GeolocationMode::GEOLOCATION_OFF)
        {
            m_lock = _lock;
            m_location = _location;
        }

        length_type size()
        {
            return 2*sizeof(byte);
        }

        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            _result.clear();
            _result.push_back(m_lock);
            _result.push_back(m_location);
            return _result;
        }

        LockMode* parse(std::vector<byte> _data, index_type _index = 0)
        {
            m_lock = _data[_index];
            m_location = _data[_index + 1];
            return this;
        }

        LockMode* parse(byte* _data, index_type _index = 0)
        {
            m_lock = _data[_index];
            m_location = _data[_index + 1];
            return this;
        }
    };
};
#endif
