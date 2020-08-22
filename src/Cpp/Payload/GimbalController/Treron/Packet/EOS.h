#ifndef EOS_H
#define EOS_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
	struct Resolution
	{
	public:
		pixel_type vertical;
		pixel_type horizontal;

		Resolution()
		{
			vertical = 0;
			horizontal = 0;
		}
		Resolution(const pixel_type _vertical, const pixel_type _horizontal)
		{
			vertical = _vertical;
			horizontal = _horizontal;
		}
		Resolution(const Resolution &_resolution)
		{
			vertical = _resolution.vertical; horizontal = _resolution.horizontal;
		}

		inline Resolution& operator =(const Resolution &_resolution)
		{
			vertical = _resolution.vertical; horizontal = _resolution.horizontal;
			return *this;
		}

		inline bool operator == (const Resolution &_resolution)
		{
			return (vertical == _resolution.vertical&& horizontal == _resolution.horizontal);
		}
	};

	class EOS:public Object
	{
	public:

		EOS()
		{
			m_resolution = Resolution();
			m_fov = 1;
			sensorId();
		}

		EOS(pixel_type _vertical, pixel_type _horizontal, data_type _fov/*Fov as rad*/)
		{
			m_resolution = Resolution(_vertical, _horizontal);
			m_fov = _fov;
			sensorId();
		}
		~EOS(){};

	protected:
		Resolution m_resolution; //Pixel * Pixel
		data_type m_fov;//Rad
		byte m_sensorID;

	public:
		virtual void sensorId() = 0;
	public:

		EOS* operator=(const EOS *ptr)
		{
			m_resolution = ptr->m_resolution;
			m_fov = ptr->m_fov;
			m_sensorID = ptr->m_sensorID;
			return this;
		}

		bool operator ==(const EOS *ptr)
		{
			return (m_resolution == ptr->m_resolution && m_fov == ptr->m_fov&&m_sensorID==ptr->m_sensorID);
		}

		length_type size()
		{
			return sizeof(m_resolution.vertical) + sizeof(m_resolution.horizontal) + sizeof(m_fov)+1;
		}

		std::vector<byte> toByte()
		{
			std::vector<byte> _result;
			std::vector<byte> b_vertical, b_horizontal, b_fov;
			b_vertical = Utils::toByte<pixel_type>(m_resolution.vertical);
			b_horizontal = Utils::toByte<pixel_type>(m_resolution.horizontal);
			b_fov = Utils::toByte<data_type>(m_fov);

			_result.clear();
			_result.push_back(m_sensorID);
			_result.insert(_result.end(), b_vertical.begin(), b_vertical.end());
			_result.insert(_result.end(), b_horizontal.begin(), b_horizontal.end());
			_result.insert(_result.end(), b_fov.begin(), b_fov.end());
			return _result;
		}

		EOS* parse(std::vector<byte> _data, index_type _index = 0)
		{
			byte* data = _data.data();
			m_sensorID = data[_index];
			m_resolution.vertical = Utils::toValue<pixel_type>(data, 1+_index);
			m_resolution.horizontal = Utils::toValue<pixel_type>(data, 1+_index + sizeof(pixel_type));
			m_fov = Utils::toValue<data_type>(data, 1+_index + 2*sizeof(pixel_type));
			return this;
		}

		EOS* parse(byte* _data, index_type _index = 0)
		{
			m_sensorID = _data[_index];
			m_resolution.vertical = Utils::toValue<pixel_type>(_data, 1 + _index);
			m_resolution.horizontal = Utils::toValue<pixel_type>(_data, 1 + _index + sizeof(pixel_type));
			m_fov = Utils::toValue<data_type>(_data, 1 + _index + 2 * sizeof(pixel_type));
			return this;
		}

	public:

		void setResolution(const Resolution &_resolution)
		{
			m_resolution = _resolution;
		}

		Resolution getResolution()const
		{
			return m_resolution;
		}

		void setFov(const data_type &_fov) //Fov at Radian
		{
			m_fov = _fov;
		}

		data_type getFov()const //Fov at radian
		{
			return m_fov;
		}

		void setFovAsDeg(const data_type _fovAsDeg)
		{
			m_fov = _fovAsDeg * 180 / 3.14159265359;
		}

		data_type getFovAsDeg()const 
		{
			return m_fov*3.14159265359 / 180;
		}

		inline byte getSensorId()const
		{
			return m_sensorID;
		}
	};
}

#endif
