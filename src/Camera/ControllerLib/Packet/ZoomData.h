#ifndef ZOOM_DATA_H
#define ZOOM_DATA_H

#include "Common_type.h"
#include "utils.h"
#include "Object.h"


#define ZOOM_IN 1
#define ZOOM_OUT 2
#define ZOOM_STOP 0
#define ZOOM_FACTOR 3

namespace Eye
{
	class ZoomData :public Object
	{
	public:
		ZoomData() :m_mode(ZOOM_STOP)
		{
			m_factor = 1;
		}

		ZoomData(const byte _mode, data_type _factor)
		{
			m_mode = _mode;
			m_factor = _factor;
		}
        ~ZoomData(){}

		inline void setZoomData(const byte _mode, data_type _factor)
		{
			m_mode = _mode;
			m_factor = _factor;
		}

		inline byte getZoomMode()const
		{
			return m_mode;
		}

		inline data_type getZoomFactor()const
		{
			return m_factor;
		}

	public:
		length_type size()
		{
			return sizeof(m_factor) + sizeof(m_mode);
		}

		std::vector<byte> toByte()
		{
            std::vector<byte> _result;
            std::vector<byte> b_factor;
            b_factor = Utils::toByte<data_type>(m_factor);
			_result.clear();
			_result.push_back(m_mode);
			_result.insert(_result.end(),b_factor.begin(),b_factor.end());
			return _result;
		}

		ZoomData* parse(std::vector<byte> _data, index_type _index = 0)
		{
            byte* data = _data.data();
            m_mode = data[_index];
            m_factor = Utils::toValue<data_type>(data,_index + sizeof(m_mode));
			return this;
		}

		ZoomData* parse(byte* _data, index_type _index = 0)
		{
            m_mode = _data[_index];
            m_factor = Utils::toValue<data_type>(_data,_index + sizeof(m_mode));
			return this;
		}

	private:
		byte m_mode;
		data_type m_factor;
	};
}

#endif
