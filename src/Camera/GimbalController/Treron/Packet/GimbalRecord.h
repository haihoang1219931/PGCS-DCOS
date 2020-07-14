#ifndef GIMBAL_RECORD_H
#define GIMBAL_RECORD_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

#define RECORD_FULL 1 //all sensor on IPC
#define RECORD_OFF 0
#define RECORD_VIEW 2
#define RECORD_TYPE_A 3
#define RECORD_TYPE_B 4

namespace Eye
{
	class GimbalRecord :public Object
	{
	public:
        GimbalRecord(){}
		GimbalRecord(const byte _mode)
		{
		}
        ~GimbalRecord(){}

        inline byte getMode()const
        {
            return m_mode;
        }

        inline void setMode(const byte _mode)
        {
            m_mode = _mode;
        }

        inline byte getType()const
        {
            return m_type;
        }

        inline void setType(const byte _type)
        {
            m_type = _type;
        }

        inline index_type getFrameID()const
        {
            return m_frameId;
        }

        inline void setFrameID(const index_type _frameId)
        {
            m_frameId = _frameId;
        }
        inline GimbalRecord& operator =(const GimbalRecord& _mode)
        {
            m_mode = _mode.m_mode;
            m_type = _mode.m_type;
            m_frameId = _mode.m_frameId;
            return *this;
        }

        inline bool operator ==(const GimbalRecord& _mode)
        {
            return (m_mode == _mode.m_mode &&
                    m_type == _mode.m_type &&
                    m_frameId == _mode.m_frameId);
        }
        length_type size()
        {
            return sizeof(m_mode) + sizeof(m_type) + sizeof(m_frameId);
        }
        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            std::vector<byte> b_frameId;
            b_frameId = Utils::toByte<index_type>(m_frameId);
            _result.clear();
            _result.push_back(m_mode);
            _result.insert(_result.end(),b_frameId.begin(), b_frameId.end());
            return _result;
        }
        GimbalRecord* parse(std::vector<byte> _data, index_type _index = 0)
        {
            byte* data = _data.data();
            m_mode = data[_index];
            m_type = data[_index+sizeof(m_mode)];
            m_frameId = Utils::toValue<index_type>(data, _index + sizeof(m_mode)+sizeof(m_type));
            return this;
        }
        GimbalRecord* parse(byte* _data, index_type _index = 0)
        {
            m_mode = _data[_index];
            m_type = _data[_index+sizeof(m_mode)];
            m_frameId = Utils::toValue<index_type>(_data, _index + sizeof(m_mode)+sizeof(m_type));
            return this;
        }
	private:
		byte m_mode;
		byte m_type;
		index_type m_frameId;

	};
}
#endif
