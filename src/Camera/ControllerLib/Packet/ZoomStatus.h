#ifndef ZOOM_STATUS_H
#define ZOOM_STATUS_H

#include "Common_type.h"
#include "utils.h"
#include "Object.h"
#include "EOS.h"


namespace Eye
{
	class ZoomStatus :public Object
	{
	public:
		ZoomStatus()
		{
		}
        ~ZoomStatus(){}

		inline void setZoomStatus(const EOS *ptr, const data_type _factor)
		{
			//TODO to calculate hfov, vfov, factor;
		}
        inline data_type sethfov(const data_type value)
        {
            m_hfov = value;
        }

        inline data_type setvfov(const data_type value)
        {
            m_vfov = value;
        }

        inline data_type setZoomFactor(const data_type value)
        {
            m_factor = value;
        }
		inline data_type gethfov()const
		{
			return m_hfov;
		}

		inline data_type getvfov()const
		{
			return m_vfov;
		}

		inline data_type getZoomFactor()const
		{
			return m_factor;
		}

	public:
		length_type size()
		{
            return sizeof(m_factor) +
                   sizeof(m_hfov) +
                   sizeof(m_vfov) ;
		}

		std::vector<byte> toByte()
		{
            std::vector<byte> _result;
            std::vector<byte> b_factor,b_hfov,b_vfov;
            b_factor = Utils::toByte<data_type>(m_factor);
            b_hfov = Utils::toByte<data_type>(m_hfov);
            b_vfov = Utils::toByte<data_type>(m_vfov);
            _result.clear();
            _result.insert(_result.end(),b_factor.begin(),b_factor.end());
            _result.insert(_result.end(),b_hfov.begin(),b_hfov.end());
            _result.insert(_result.end(),b_vfov.begin(),b_vfov.end());
            return _result;
		}

		ZoomStatus* parse(std::vector<byte> _data, index_type _index = 0)
		{
            byte* data = _data.data();
            m_factor = Utils::toValue<data_type>(data,_index );
            m_hfov = Utils::toValue<data_type>(data,_index + sizeof(m_factor));
            m_vfov = Utils::toValue<data_type>(data,_index + sizeof(m_factor) + sizeof(m_hfov));
			return this;
		}

		ZoomStatus* parse(byte* _data, index_type _index = 0)
		{
            m_factor = Utils::toValue<data_type>(_data,_index );
            m_hfov = Utils::toValue<data_type>(_data,_index + sizeof(m_factor));
            m_vfov = Utils::toValue<data_type>(_data,_index + sizeof(m_factor) + sizeof(m_hfov));
			return this;
		}

	private:
		data_type m_factor;
		data_type m_hfov;
		data_type m_vfov;
	};
}

#endif
