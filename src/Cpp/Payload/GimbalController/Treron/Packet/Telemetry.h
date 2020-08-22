#ifndef TELEMETRY_H
#define TELEMETRY_H

#include "Common_type.h"
#include "utils.h"
#include "Object.h"

namespace  Eye {
    class Telemetry :public Object
    {
    private:
        data_type m_pn;
        data_type m_pe;
        data_type m_pd;
        data_type m_roll;
        data_type m_pitch;
        data_type m_yaw;
        data_type m_speedNorth;
        data_type m_speedEast;
        data_type m_takeOffAlt;
        data_type m_gpsAlt;

    public:
        Telemetry(){
            m_pn = 0.0;
            m_pe = 0.0;
            m_pd = 0.0;
            m_roll = 0.0;
            m_pitch = 0.0;
            m_yaw = 0.0;
            m_speedNorth = 0.0;
            m_speedEast = 0.0;
            m_takeOffAlt = 0.0;
            m_gpsAlt = 0.0;
        }

        Telemetry(const data_type _pn, const data_type _pe, const data_type _pd,
                  const data_type _roll, const data_type _pitch, const data_type _yaw,
                  const data_type _speedNorth, data_type _speedEast, data_type _takeOffAlt, data_type _gpsAlt)
            : m_pn(_pn), m_pe(_pe), m_pd(_pd), m_roll(_roll), m_pitch(_pitch), m_yaw(_yaw),
              m_speedNorth(_speedNorth), m_speedEast(_speedEast), m_takeOffAlt(_takeOffAlt), m_gpsAlt(_gpsAlt)
        {

        }

        Telemetry(const Telemetry& _telemetry){
            m_pn = _telemetry.m_pn;
            m_pe = _telemetry.m_pe;
            m_pd = _telemetry.m_pd;
            m_roll = _telemetry.m_roll;
            m_pitch = _telemetry.m_pitch;
            m_yaw = _telemetry.m_yaw;
            m_speedNorth = _telemetry.m_speedNorth;
            m_speedEast = _telemetry.m_speedEast;
            m_gpsAlt = _telemetry.m_gpsAlt;
            m_takeOffAlt = _telemetry.m_takeOffAlt;
        }

        ~Telemetry(){}

        data_type getPn()const{return m_pn;}
        void setPn(const data_type _pn){m_pn = _pn;}
        data_type getPe()const{return m_pe;}
        void setPe(const data_type _pe){m_pe = _pe;}
        data_type getPd()const{return m_pd; }
        void setPd(const data_type _pd){m_pd = _pd;}
        data_type getRoll(){return m_roll;}
        data_type getPitch(){return m_pitch;}
        data_type getYaw(){return m_yaw;}
        void setRoll(data_type _roll){m_roll = _roll;}
        void setPitch(data_type _pitch){m_pitch = _pitch;}
        void setYaw(data_type _yaw){m_yaw = _yaw;}
        data_type getSpeedNorth(){return m_speedNorth;}
        data_type getSpeedEast(){return m_speedEast;}
        data_type getGPSAlt(){return m_gpsAlt;}
        data_type getTakeOffAlt(){return m_takeOffAlt;}
        void setSpeedNorth(data_type _speedNorth){ m_speedNorth = _speedNorth; }
        void setSpeedEast(data_type _speedEast){ m_speedEast = _speedEast; }
        void setGPSAlt(data_type _gpsAlt){ m_gpsAlt = _gpsAlt; }
        void setTakeOffAlt(data_type _takeOffAlt){ m_takeOffAlt = _takeOffAlt; }

        Telemetry & operator=(const Telemetry &_telemetry)
        {
            m_pd = _telemetry.m_pd;
            m_pn = _telemetry.m_pn;
            m_pe = _telemetry.m_pe;
            m_roll = _telemetry.m_roll;
            m_pitch = _telemetry.m_pitch;
            m_yaw = _telemetry.m_yaw;
            m_speedNorth = _telemetry.m_speedNorth;
            m_speedEast = _telemetry.m_speedEast;
            m_gpsAlt = _telemetry.m_gpsAlt;
            m_takeOffAlt = _telemetry.m_takeOffAlt;
            return *this;
        }

        bool operator==(const Telemetry &_telemetry){
            return  m_pd == _telemetry.m_pd &&
                    m_pn == _telemetry.m_pn &&
                    m_pe == _telemetry.m_pe &&
                    m_roll == _telemetry.m_roll &&
                    m_pitch == _telemetry.m_pitch &&
                    m_yaw == _telemetry.m_yaw&&
                    m_speedNorth == _telemetry.m_speedNorth&&
                    m_speedEast == _telemetry.m_speedEast&&
                    m_gpsAlt == _telemetry.m_gpsAlt&&
                    m_takeOffAlt == _telemetry.m_takeOffAlt;
        }

        length_type size(){return  sizeof(data_type) * 10;}

        std::vector<byte> toByte(){
            std::vector<byte> _result;
            std::vector<unsigned char> b_roll, b_pitch, b_yaw, b_pn, b_pe, b_pd, b_speedNorth, b_speedEast, b_gpsAlt, b_takeOffAlt;
            b_pn = Utils::toByte<data_type>(m_pn);
            b_pe = Utils::toByte<data_type>(m_pe);
            b_pd = Utils::toByte<data_type>(m_pd);
            b_roll = Utils::toByte<data_type>(m_roll);
            b_pitch = Utils::toByte<data_type>(m_pitch);
            b_yaw = Utils::toByte<data_type>(m_yaw);
            b_speedNorth = Utils::toByte<data_type>(m_speedNorth);
            b_speedEast = Utils::toByte<data_type>(m_speedEast);
            b_gpsAlt = Utils::toByte<data_type>(m_gpsAlt);
            b_takeOffAlt = Utils::toByte<data_type>(m_takeOffAlt);


            _result = b_pn;
            _result.insert(_result.end(), b_pe.begin(), b_pe.end());
            _result.insert(_result.end(), b_pd.begin(), b_pd.end());
            _result.insert(_result.end(), b_roll.begin(), b_roll.end());
            _result.insert(_result.end(), b_pitch.begin(), b_pitch.end());
            _result.insert(_result.end(), b_yaw.begin(), b_yaw.end());
            _result.insert(_result.end(), b_speedNorth.begin(), b_speedNorth.end());
            _result.insert(_result.end(), b_speedEast.begin(), b_speedEast.end());
            _result.insert(_result.end(), b_gpsAlt.begin(), b_gpsAlt.end());
            _result.insert(_result.end(), b_takeOffAlt.begin(), b_takeOffAlt.end());
            return _result;
        }

        Telemetry* parse(std::vector<byte> _data, index_type _index = 0){
            byte* data = _data.data();
            m_pn = Utils::toValue<data_type>(data, _index);
            m_pe = Utils::toValue<data_type>(data, _index + sizeof(data_type));
            m_pd = Utils::toValue<data_type>(data, _index + 2 * sizeof(data_type));
            m_roll = Utils::toValue<data_type>(data, _index + 3 * sizeof(data_type));
            m_pitch = Utils::toValue<data_type>(data, _index + 4 * sizeof(data_type));
            m_yaw = Utils::toValue<data_type>(data, _index + 5 * sizeof(data_type));
            m_speedNorth = Utils::toValue<data_type>(data, _index + 6 * sizeof(data_type));
            m_speedEast = Utils::toValue<data_type>(data, _index + 7 * sizeof(data_type));
            m_gpsAlt = Utils::toValue<data_type>(data, _index + 8 * sizeof(data_type));
            m_takeOffAlt = Utils::toValue<data_type>(data, _index + 9 * sizeof(data_type));
            return this;
        }

        Telemetry* parse(byte* _data, index_type _index = 0){
            m_pn = Utils::toValue<data_type>(_data, _index);
            m_pe = Utils::toValue<data_type>(_data, _index + sizeof(data_type));
            m_pd = Utils::toValue<data_type>(_data, _index + 2 * sizeof(data_type));
            m_roll = Utils::toValue<data_type>(_data, _index + 3 * sizeof(data_type));
            m_pitch = Utils::toValue<data_type>(_data, _index + 4 * sizeof(data_type));
            m_yaw = Utils::toValue<data_type>(_data, _index + 5 * sizeof(data_type));
            m_speedNorth = Utils::toValue<data_type>(_data, _index + 6 * sizeof(data_type));
            m_speedEast = Utils::toValue<data_type>(_data, _index + 7 * sizeof(data_type));
            m_gpsAlt = Utils::toValue<data_type>(_data, _index + 8 * sizeof(data_type));
            m_takeOffAlt = Utils::toValue<data_type>(_data, _index + 9 * sizeof(data_type));
        }
    };

}
#endif // TELEMETRY_H
