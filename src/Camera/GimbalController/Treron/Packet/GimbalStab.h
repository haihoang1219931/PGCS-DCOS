#ifndef GIMBAL_STAB_H
#define GIMBAL_STAB_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

#define GIMBAL_PAN_ON 1
#define GIMBAL_PAN_OFF 0
#define GIMBAL_TILT_ON 1
#define GIMBAL_TILT_OFF 0

namespace Eye
{
	class GimbalStab :public Object
	{
	public:
		GimbalStab(){}
		~GimbalStab(){}
		inline byte getStabPan()const
		{
			return m_stabPan;
		}

		inline void setStabPan(const byte _value)
		{
			m_stabPan = _value;
		}

        inline byte getStabTilt()const
		{
			return m_stabTilt;
		}

		inline void setStabTilt(const byte _value)
		{
			m_stabTilt = _value;
		}

		inline GimbalStab& operator =(const GimbalStab& _stabMode)
		{
			m_stabPan = _stabMode.m_stabPan;
			m_stabTilt = _stabMode.m_stabTilt;
			return *this;
		}

		inline bool operator ==(const GimbalStab& _stabMode)
		{
			return (m_stabPan == _stabMode.m_stabPan&&
                    m_stabTilt == _stabMode.m_stabTilt);
		}

		length_type size()
		{
			return sizeof(m_stabPan) + sizeof(m_stabTilt);
		}

		std::vector<byte> toByte()
		{
			std::vector<byte> _result;
			_result.clear();
			_result.push_back(m_stabPan);
			_result.push_back(m_stabTilt);
			return _result;
		}
		GimbalStab* parse(std::vector<byte> _data, index_type _index = 0)
		{
			m_stabPan = _data[_index];
			m_stabTilt = _data[_index + sizeof(m_stabPan)];
			return this;
		}

		GimbalStab* parse(unsigned char* _data, index_type _index = 0)
		{
			m_stabPan = _data[_index];
			m_stabTilt = _data[_index + sizeof(m_stabPan)];
			return this;
		}
	private:
		byte m_stabPan;
		byte m_stabTilt;
	};
}
#endif
