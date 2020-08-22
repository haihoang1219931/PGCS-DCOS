#ifndef GPS_DATA_H
#define GPS_DATA_H
#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
	class GPSData :public Object
	{
	private:
		data_type m_pn;
		data_type m_pe;
		data_type m_pd;
	public:
		GPSData()
		{
			m_pn = 0;
			m_pe = 0;
			m_pd = 0;
		}

		GPSData(const data_type _pn, const data_type _pe, const data_type _pd)
		{
			m_pn = _pn;
			m_pe = _pe;
			m_pd = _pd;
		}

		GPSData(const GPSData& _location)
		{
			m_pn = _location.m_pn;
			m_pe = _location.m_pe;
			m_pd = _location.m_pd;
		}
		~GPSData(){}

		inline data_type getPn()const
		{
			return m_pn;
		}

		inline void setPn(const data_type _pn)
		{
			m_pn = _pn;
		}


		inline data_type getPe()const
		{
			return m_pe;
		}

		inline void setPe(const data_type _pe)
		{
			m_pe = _pe;
		}

		inline data_type getPd()const
		{
			return m_pd;
		}

		inline void setPd(const data_type _pd)
		{
			m_pd = _pd;
		}

		inline GPSData & operator=(const GPSData &_location)
		{
            m_pd = _location.m_pd;
			m_pn = _location.m_pn;
			m_pe = _location.m_pe;
			return *this;
		}
        inline bool operator==(const GPSData &_location){
            return  m_pd == _location.m_pd &&
                    m_pn == _location.m_pn &&
                    m_pe == _location.m_pe;
        }
		length_type size()
		{
			return sizeof(data_type)* 3;
		}

		std::vector<byte> toByte()
		{
			std::vector<byte> _result;

			std::vector<unsigned char> b_pn, b_pe, b_pd;
			b_pn = Utils::toByte<data_type>(m_pn);
			b_pe = Utils::toByte<data_type>(m_pe);
			b_pd = Utils::toByte<data_type>(m_pd);

            _result = b_pn;
			_result.insert(_result.end(), b_pe.begin(), b_pe.end());
			_result.insert(_result.end(), b_pd.begin(), b_pd.end());
			return _result;
		}

		GPSData* parse(std::vector<byte> _data, index_type _index = 0)
		{
			byte* data = _data.data();
			m_pn = Utils::toValue<data_type>(data, _index);
			m_pe = Utils::toValue<data_type>(data, _index + sizeof(data_type));
			m_pd = Utils::toValue<data_type>(data, _index + 2 * sizeof(data_type));
			return this;
		}

		GPSData* parse(unsigned char* _data, index_type _index = 0)
		{
			m_pn = Utils::toValue<data_type>(_data, _index);
			m_pe = Utils::toValue<data_type>(_data, _index + sizeof(data_type));
			m_pd = Utils::toValue<data_type>(_data, _index + 2 * sizeof(data_type));
			return this;
		}

	};
}


#endif
