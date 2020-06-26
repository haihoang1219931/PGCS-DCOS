#ifndef PT_ANGLE_DIFF_H
#define PT_ANGLE_DIFF_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
	class PTAngleDiff: public Object
	{
	private:
		data_type m_panAngleDiff;
		data_type m_tiltAngleDiff;
	public:
		PTAngleDiff(){
            m_panAngleDiff = 0;
            m_tiltAngleDiff = 0;
		}
		PTAngleDiff(const data_type &_panDiff, const data_type &_tiltDiff)
		{
			m_panAngleDiff = _panDiff;
			m_tiltAngleDiff = _tiltDiff;
		}
		~PTAngleDiff(){}
	public:
        inline void setPanAngleDiff(const data_type &_panDiff)
		{
			m_panAngleDiff = _panDiff;
		}
		inline void setTiltAngleDiff(const data_type &_tiltDiff)
		{
			m_tiltAngleDiff = _tiltDiff;
		}
		inline void setPTAngleDiff(const data_type &_panDiff, const data_type &_tiltDiff)
		{
			m_panAngleDiff = _panDiff;
			m_tiltAngleDiff = _tiltDiff;
		}
		inline data_type getPanAngleDiff()const
		{
			return m_panAngleDiff;
		}
		inline data_type getTiltAngleDiff() const
		{
			return m_tiltAngleDiff;
		}
		inline PTAngleDiff & operator=(const PTAngleDiff &_angleDiff)
		{
			m_panAngleDiff = _angleDiff.getPanAngleDiff();
			m_tiltAngleDiff = _angleDiff.getTiltAngleDiff();
			return *this;
		}

		inline bool operator==(const PTAngleDiff &_angleDiff)
		{
			return (
                    m_panAngleDiff == _angleDiff.getPanAngleDiff() &&
                    m_tiltAngleDiff == _angleDiff.getTiltAngleDiff()
                    );
		}
		length_type size()
		{
			return sizeof(m_panAngleDiff) + sizeof(m_tiltAngleDiff);
		}
        std::vector<byte> toByte()
		{
			std::vector<byte> _result;
			std::vector<byte> b_panAngleDiff,b_tiltAngleDiff;
			b_panAngleDiff = Utils::toByte<data_type>(m_panAngleDiff);
			b_tiltAngleDiff = Utils::toByte<data_type>(m_tiltAngleDiff);

			_result.clear();
			_result.insert(_result.end(),b_panAngleDiff.begin(),b_panAngleDiff.end());
			_result.insert(_result.end(),b_tiltAngleDiff.begin(),b_tiltAngleDiff.end());
			return _result;
		}
		PTAngleDiff* parse(std::vector<byte> _data, index_type _index = 0)
		{
            byte* data = _data.data();
			m_panAngleDiff =  Utils::toValue<data_type>(data,_index);
			m_tiltAngleDiff = Utils::toValue<data_type>(data,_index+sizeof(m_panAngleDiff));
			return this;
		}
		PTAngleDiff* parse(byte* _data, index_type _index = 0)
		{
			m_panAngleDiff =  Utils::toValue<data_type>(_data,_index);
			m_tiltAngleDiff = Utils::toValue<data_type>(_data,_index+sizeof(m_panAngleDiff));
			return this;
		}

	};
}
#endif
