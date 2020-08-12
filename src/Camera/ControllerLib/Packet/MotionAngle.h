#ifndef MOTION_ANGLE_H
#define MOTION_ANGLE_H
#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
	//All data of motionC packet to calculate EyeModel
	//This Object used on MotionC, EyeMain
	// Transfer from MotionC -> Other and Other (Not include MotionC)
	class MotionAngle :public Object
	{
	private:
		data_type m_angle; // Angle of axis, measured by Encoder - using for Motion Control
		data_type m_ref_enc; // Reference angle of axis, for config exactly angle
		data_type m_maxRate; //
	public:
		MotionAngle()
		{
			m_angle = 0;
			m_ref_enc = 0;
			m_maxRate = 0;
		}

		MotionAngle(data_type _angle,data_type _maxRate = 0, data_type _refEnc = 0)
		{
			m_ref_enc = _refEnc;
			m_angle = _angle - m_ref_enc;
			m_maxRate = _maxRate;
		}

		MotionAngle(const MotionAngle &_motionAngle)
		{
			m_angle = _motionAngle.m_angle;
			m_ref_enc = _motionAngle.m_ref_enc;
			m_maxRate = _motionAngle.m_maxRate;
		}

		~MotionAngle(){}

		inline data_type getAngleEnc() const// Angle on Encoder
		{
			return m_angle;
		}

		inline data_type getAngleGeo()const //Angle on geometry
		{
			return m_angle - m_ref_enc;
		}

		inline void setAngleGeo(const data_type _angle) // _angle base on Geometry
		{
			m_angle = _angle + m_ref_enc;
		}

		inline void setAngleEnc(const data_type _angle) // _angle base on Geometry
		{
			m_angle = _angle;
		}

		inline data_type getRefEnc()const
		{
			return m_ref_enc;
		}

		inline void setRefEnc(const data_type _ref_enc)
		{
			m_ref_enc = _ref_enc;
		}

		inline data_type getMaxRate()const
		{
			return m_maxRate;
		}

		inline void setMaxRate(const data_type &_max_rate)
		{
			m_maxRate = _max_rate;
		}


		inline MotionAngle& operator=(const MotionAngle& _motionAngle)
		{
			m_angle = _motionAngle.m_angle;
			m_ref_enc = _motionAngle.m_ref_enc;
			m_maxRate = _motionAngle.m_maxRate;
			return *this;
		}

		inline bool operator == (const MotionAngle& _motionC)
		{
			return (m_angle == _motionC.m_angle&&m_ref_enc == _motionC.m_ref_enc);
		}

        length_type size()
		{
			return sizeof(data_type)* 3;
		}

		std::vector<byte> toByte()
		{
			std::vector<byte> _result(0);
			std::vector<byte> b_angle, b_maxRate, b_ref_enc;
			b_angle = Utils::toByte<data_type>(m_angle);
			b_maxRate = Utils::toByte<data_type>(m_maxRate);
			b_ref_enc = Utils::toByte<data_type>(m_ref_enc);
			_result = b_angle;
			_result.insert(_result.end(), b_maxRate.begin(), b_maxRate.end());
			_result.insert(_result.end(), b_ref_enc.begin(), b_ref_enc.end());
			return _result;
		}

		MotionAngle* parse(std::vector<byte> _data, index_type _index = 0)
		{
			byte* data = _data.data();
			m_angle = Utils::toValue<data_type>(data, _index);
			m_maxRate = Utils::toValue<data_type>(data, _index + sizeof(data_type));
			m_ref_enc = Utils::toValue<data_type>(data, _index + 2 * sizeof(data_type));
			return this;
		}

		MotionAngle* parse(byte* _data, index_type _index = 0)
		{
			m_angle = Utils::toValue<data_type>(_data, _index);
			m_maxRate = Utils::toValue<data_type>(_data, _index + sizeof(data_type));
			m_ref_enc = Utils::toValue<data_type>(_data, _index + 2 * sizeof(data_type));
			return this;
		}
	};
}


#endif
