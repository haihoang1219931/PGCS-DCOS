#ifndef PT_ANGLE__H
#define PT_ANGLE__H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
    class PTAngle: public Object
	{
	private:
        data_type m_panAngle;
        data_type m_tiltAngle;
	public:
        PTAngle(){
            m_panAngle = 0;
            m_tiltAngle = 0;
		}
        PTAngle(const data_type &_pan, const data_type &_tilt)
		{
            m_panAngle = _pan;
            m_tiltAngle = _tilt;
		}
        ~PTAngle(){}
	public:
        inline void setPanAngle(const data_type &_pan)
		{
            m_panAngle = _pan;
		}
        inline void setTiltAngle(const data_type &_tilt)
		{
            m_tiltAngle = _tilt;
		}
        inline void setPTAngle(const data_type &_pan, const data_type &_tilt)
		{
            m_panAngle = _pan;
            m_tiltAngle = _tilt;
		}
        inline data_type getPanAngle()const
		{
            return m_panAngle;
		}
        inline data_type getTiltAngle() const
		{
            return m_tiltAngle;
		}
        inline PTAngle & operator=(const PTAngle &_angle)
		{
            m_panAngle = _angle.getPanAngle();
            m_tiltAngle = _angle.getTiltAngle();
			return *this;
		}

        inline bool operator==(const PTAngle &_angle)
		{
			return (
                    m_panAngle == _angle.getPanAngle() &&
                    m_tiltAngle == _angle.getTiltAngle()
                    );
		}
		length_type size()
		{
            return sizeof(m_panAngle) + sizeof(m_tiltAngle);
		}
        std::vector<byte> toByte()
		{
			std::vector<byte> _result;
            std::vector<byte> b_panAngle,b_tiltAngle;
            b_panAngle = Utils::toByte<data_type>(m_panAngle);
            b_tiltAngle = Utils::toByte<data_type>(m_tiltAngle);

			_result.clear();
            _result.insert(_result.end(),b_panAngle.begin(),b_panAngle.end());
            _result.insert(_result.end(),b_tiltAngle.begin(),b_tiltAngle.end());
			return _result;
		}
        PTAngle* parse(std::vector<byte> _data, index_type _index = 0)
		{
            byte* data = _data.data();
            m_panAngle =  Utils::toValue<data_type>(data,_index);
            m_tiltAngle = Utils::toValue<data_type>(data,_index+sizeof(m_panAngle));
			return this;
		}
        PTAngle* parse(byte* _data, index_type _index = 0)
		{
            m_panAngle =  Utils::toValue<data_type>(_data,_index);
            m_tiltAngle = Utils::toValue<data_type>(_data,_index+sizeof(m_panAngle));
			return this;
		}

	};
}
#endif
