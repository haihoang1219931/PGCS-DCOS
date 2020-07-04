#ifndef EYE_EVENT_H
#define EYE_EVENT_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

#define MEMORY_FULL 1
#define CHECK_MOTION 2
#define CHECK_IPC 3
#define CHECK_OFF 0

namespace Eye
{
	//Need Log file
	//Process if have event

	class EyeEvent: public Object
	{
    public:
        EyeEvent(){}
        EyeEvent(const byte _eventId, const byte _value)
		{
            m_eventId = _eventId;
            m_value = _value;
		}
        ~EyeEvent(){}
        inline byte getEventID()const
        {
            return m_eventId;
        }

        inline void setEventID(const byte _eventId)
        {
            m_eventId = _eventId;
        }

        inline byte getValue()const
        {
            return m_value;
        }

        inline void setType(const byte _value)
        {
            m_value = _value;
        }

        inline EyeEvent& operator =(const EyeEvent& _mode)
        {
            m_eventId = _mode.m_eventId;
            m_value = _mode.m_value;
            return *this;
        }

        inline bool operator ==(const EyeEvent& _mode)
        {
            return (m_eventId == _mode.m_eventId &&
                    m_value == _mode.m_value);
        }
        length_type size()
        {
            return sizeof(m_eventId) + sizeof(m_value);
        }
        std::vector<byte> toByte()
        {
            std::vector<byte> _result;           
            _result.clear();
            _result.push_back(m_eventId);
            _result.push_back(m_value);
            return _result;
        }
        EyeEvent* parse(std::vector<byte> _data, index_type _index = 0)
        {
            byte* data = _data.data();
            m_eventId = data[_index];
            m_value = data[_index+sizeof(m_eventId)];
            return this;
        }
        EyeEvent* parse(byte* _data, index_type _index = 0)
        {
            m_eventId = _data[_index];
            m_value = _data[_index+sizeof(m_eventId)];
            return this;
        }
	private:
		byte m_eventId;
		byte m_value;
	};
}
#endif
