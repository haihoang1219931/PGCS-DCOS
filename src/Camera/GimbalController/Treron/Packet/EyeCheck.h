#ifndef EYE_CHECK_H
#define EYE_CHECK_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

#define CHECK_FULL 1
#define CHECK_MOTION 2
#define CHECK_IPC 3
#define CHECK_OFF 0

namespace Eye
{
	class EyeCheck :public Object
	{
	public:
		EyeCheck(){};
		EyeCheck(const byte _mode)
		{
            m_chkMode = _mode;
		}
		~EyeCheck(){}
		inline byte getMode() const
		{
			return m_chkMode;
		}

		inline void setMode(const byte &_mode)
		{
			m_chkMode = _mode;
		}

		inline EyeCheck& operator =(const EyeCheck& _mode)
		{
			m_chkMode = _mode.m_chkMode;
			return *this;
		}

        inline bool operator ==(const EyeCheck& _mode)
		{
			return (m_chkMode == _mode.m_chkMode);
		}

		length_type size()
		{
			return sizeof(m_chkMode);
		}

		std::vector<byte> toByte()
		{
			std::vector<byte> _result;
			_result.clear();
			_result.push_back(m_chkMode);
			return _result;
		}
        EyeCheck* parse(std::vector<byte> _data, index_type _index = 0)
		{
			m_chkMode = _data[_index];
			return this;
		}

        EyeCheck* parse(unsigned char* _data, index_type _index = 0)
		{
			m_chkMode = _data[_index];
			return this;
		}
	private:
		byte m_chkMode;
	};
}
#endif
