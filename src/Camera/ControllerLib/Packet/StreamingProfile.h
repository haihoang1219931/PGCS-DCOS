#ifndef STREAMING_PROFILE_H
#define STREAMING_PROFILE_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"
#include "EyeStatus.h"

namespace Eye
{
    class StreamingProfile :public Object
    {
    public:
        StreamingProfile(){}
        ~StreamingProfile(){}
        inline byte getProfile()const
        {
            return m_profile;
        }

        inline void setProfile(const byte _value)
        {
            m_profile = _value;
        }


        inline StreamingProfile& operator =(const StreamingProfile& _value)
        {
            m_profile = _value.m_profile;
            return *this;
        }

        inline bool operator ==(const StreamingProfile& _value)
        {
            return (m_profile == _value.m_profile);
        }

        length_type size()
        {
            return sizeof(m_profile);
        }

        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            _result.push_back(m_profile);
            return _result;
        }
        StreamingProfile* parse(std::vector<byte> _data, index_type _index = 0)
        {
            m_profile = _data[_index];
            return this;
        }

        StreamingProfile* parse(unsigned char* _data, index_type _index = 0)
        {
            m_profile = _data[_index];
            return this;
        }
    private:
        byte m_profile;
    };
}
#endif
