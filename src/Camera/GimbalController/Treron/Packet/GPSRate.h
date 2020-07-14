#ifndef GPSRATE_H
#define GPSRATE_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
	class GPSRate :public Object
	{
	public:
		GPSRate()
		{
			m_v = 0;
			m_x = 0;
		}

		GPSRate(const data_type _v, const data_type _x)
		{
			m_v = _v;
			m_x = _x;
		}

		GPSRate(const GPSRate& _gpsRate)
		{
			m_v = _gpsRate.m_v;
			m_x = _gpsRate.m_x;
		}
		~GPSRate(){}

		inline data_type getX()const
		{
			return m_x;
		}

		inline void setX(const data_type _x)
		{
			m_x = _x;
		}

		inline data_type getV()const
		{
			return m_v;
		}

		inline void setV(const data_type _v)
		{
			m_v = _v;
		}

		inline GPSRate & operator=(const GPSRate &_gpsRate)
		{
			m_v = _gpsRate.m_v;
			m_x = _gpsRate.m_x;
			return *this;
		}

		length_type size()
		{
			return sizeof(data_type)* 2;
		}

		std::vector<byte> toByte()
		{
			std::vector<byte> _result;

			std::vector<unsigned char> b_v, b_x;
			b_v = Utils::toByte<data_type>(m_v);
			b_x = Utils::toByte<data_type>(m_x);

			_result = b_v;
			_result.insert(_result.end(), b_x.begin(), b_x.end());
			return _result;
		}

		GPSRate* parse(std::vector<byte> _data, index_type _index = 0)
		{
			byte* data = _data.data();
			m_v = Utils::toValue<data_type>(data, _index);
			m_x = Utils::toValue<data_type>(data, _index + sizeof(data_type));
			return this;
		}

		GPSRate* parse(unsigned char* _data, index_type _index = 0)
		{
			m_v = Utils::toValue<data_type>(_data, _index);
			m_x = Utils::toValue<data_type>(_data, _index + sizeof(data_type));
			return this;
		}
	private:
		data_type m_v;
		data_type m_x;
	};
};
#endif