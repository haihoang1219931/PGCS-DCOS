#ifndef RFDATA_H
#define RFDATA_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"
#include "GPSData.h"

#define RF_LRF 1
#define RF_IPC 2

namespace Eye
{
	class RFData :public Object
	{
    private:
		byte m_dMode; //distance mode
		data_type m_distance; //Base GEO
		GPSData m_gpsTarget;
		GPSData m_gpsUAV;
	public:
        RFData(){}
        ~RFData(){}
		//
        inline byte getDMode() const{
            return m_dMode;
        }
        inline void setMode(const byte _dMode){
            m_dMode = _dMode;
        }

        inline data_type getDistance() const{
            return m_distance;
        }
        inline void setDistance(const data_type _distance){
            m_distance = _distance;
        }
        inline GPSData getGPSTarget() const{
            return m_gpsTarget;
        }
        inline void setGPSTarget(const GPSData _gcsTarget){
            m_gpsTarget = _gcsTarget;
        }
        inline GPSData getGPSUAV() const{
            return m_gpsUAV;
        }
        inline void setGPSUAV(const GPSData _gcsUAV){
            m_gpsUAV = _gcsUAV;
        }
        inline RFData & operator=(const RFData &_data)
		{
			m_dMode = _data.m_dMode;
			m_distance = _data.m_distance;
			m_gpsTarget = _data.m_gpsTarget;
			m_gpsUAV = _data.m_gpsUAV;
			return *this;
		}

		inline bool operator==(const RFData &_data)
		{
			return (m_dMode == _data.m_dMode &&
                    m_distance == _data.m_distance &&
                    m_gpsTarget == _data.m_gpsTarget &&
                    m_gpsUAV == _data.m_gpsUAV);
		}

        length_type size()
		{
			return sizeof(m_dMode) + sizeof(m_distance) + m_gpsTarget.size()+ m_gpsUAV.size();
		}

		std::vector<byte> toByte()
		{
			std::vector<byte> _result;
			std::vector<byte>b_distance,b_gpsTarget,b_gpsUAV;
			b_distance = Utils::toByte<data_type>(m_distance);
			b_gpsTarget = m_gpsTarget.toByte();
			b_gpsUAV = m_gpsUAV.toByte();

			_result.clear();
			_result.push_back(m_dMode);
			_result.insert(_result.end(),b_distance.begin(), b_distance.end());
			_result.insert(_result.end(),b_gpsTarget.begin(), b_gpsTarget.end());
			_result.insert(_result.end(),b_gpsUAV.begin(), b_gpsUAV.end());
			return _result;
		}
		RFData* parse(std::vector<byte> _data, index_type _index = 0)
		{
            //printf("Parse vector\r\n");
			m_dMode = _data[_index];
			m_distance = Utils::toValue<data_type>(_data, _index + sizeof(m_dMode));
			m_gpsTarget.parse(_data,_index + sizeof(m_dMode)+ sizeof(m_distance));
			m_gpsUAV.parse(_data,_index + sizeof(m_dMode)+ sizeof(m_distance) + m_gpsTarget.size());
			return this;
		}
		RFData* parse(byte* _data, index_type _index = 0)
		{
            //printf("Parse byte\r\n");
			m_dMode = _data[_index];
			m_distance = Utils::toValue<data_type>(_data, _index + sizeof(m_dMode));
			m_gpsTarget.parse(_data,_index + sizeof(m_dMode)+ sizeof(m_distance));
			m_gpsUAV.parse(_data,_index + sizeof(m_dMode)+ sizeof(m_distance) + m_gpsTarget.size());
			return this;
		}
		//

	};
}
#endif
