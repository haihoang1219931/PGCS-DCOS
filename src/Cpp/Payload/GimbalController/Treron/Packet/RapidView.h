#ifndef RAPID_VIEW_H
#define RAPID_VIEW_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"
#include "MotionAngle.h"
#include "InstallMode.h"
#include "EOS.h"


namespace Eye
{
	class RapidView :public Object
	{
    private:
		byte m_mode;
		//TODO
	public:
        RapidView() :m_mode(){}
		~RapidView(){}

		inline RapidView & operator=(const RapidView &_mode)
		{
			m_mode = _mode.m_mode;
			return *this;
		}

		inline bool operator==(const RapidView &_mode)
		{
			return (m_mode == _mode.m_mode);
		}

		inline byte getViewMode()const
		{
			return m_mode;
		}

		inline void setViewMode(const byte _mode)
		{
			m_mode = _mode;
		}
        length_type size()
		{
			return sizeof(m_mode);
		}
        std::vector<byte> toByte()
		{
			std::vector<byte> _result;
			_result.push_back(m_mode);
			return _result;
		}
		RapidView* parse(std::vector<byte> _data, index_type _index = 0)
		{
			m_mode = _data[_index];
			return this;
		}
		RapidView* parse(byte* _data, index_type _index = 0)
		{
			m_mode = _data[_index];
			return this;
		}

	};
}
#endif
