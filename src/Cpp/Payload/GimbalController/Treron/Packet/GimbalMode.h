#ifndef GIMBAL_MODE_H
#define GIMBAL_MODE_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
	class GimbalMode :public Object
	{
	public:
        GimbalMode(){}
        GimbalMode(const byte _mode) : m_mode(_mode)
        {
		}
        ~GimbalMode(){}
        inline void setMode(const byte _mode){
            m_mode = _mode;
        }
        inline byte getMode() const{
            return m_mode;
        }
        inline GimbalMode& operator =(const GimbalMode& _mode)
        {
            m_mode = _mode.m_mode;
            return *this;
        }

        inline bool operator ==(const GimbalMode& _mode)
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

        GimbalMode* parse(std::vector<byte> _data, index_type _index = 0)
        {
            m_mode = _data[_index];
            return this;
        }

        GimbalMode* parse(unsigned char* _data, index_type _index = 0)
        {
            m_mode = _data[_index];
            return this;
        }

	private:
		byte m_mode;
	};
}
#endif
