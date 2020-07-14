#ifndef MOTIONCSTATUS_H
#define MOTIONCSTATUS_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
    class MotionCStatus :public Object
    {
    public:
        MotionCStatus(){

        }

        MotionCStatus(byte _panStabMode,
                      byte _tiltStabMode,
                      data_type _panCurrentVelocity,
                      data_type _tiltCurrentVelocity,
                      data_type _panCurrentPosition,
                      data_type _tiltCurrentPosition)
            : m_panCurrentPosition(_panCurrentPosition),
              m_tiltCurrentPosition(_tiltCurrentPosition),
              m_panCurrentVelocity(_panCurrentVelocity),
              m_tiltCurrentVelocity(_tiltCurrentVelocity),
              m_panStabMode(_panStabMode),
              m_tiltStabMode(_tiltStabMode)
        {

        }

        ~MotionCStatus(){

        }

    public:
        length_type size(){
            return sizeof(data_type)*4 + 2*sizeof(byte);
        }

        byte getPanStabMode(){ return m_panStabMode; }
        byte getTiltStabMode(){ return m_tiltStabMode; }
        data_type getPanPos(){ return m_panCurrentPosition; }
        data_type getTiltPos(){ return m_tiltCurrentPosition; }
        data_type getPanVelo(){ return m_panCurrentVelocity; }
        data_type getTiltVelo(){ return m_tiltCurrentVelocity; }

        void setPanStabMode(byte _panStabMode){ m_panStabMode = _panStabMode; }
        void setTiltStabMode(byte _tiltStabMode){ m_tiltStabMode = _tiltStabMode; }
        void setPanPos(data_type _panPos){ m_panCurrentPosition = _panPos; }
        void setTiltPos(data_type _tiltPos){ m_tiltCurrentPosition = _tiltPos; }
        void setPanVelo(data_type _panVelo){ m_panCurrentVelocity = _panVelo; }
        void setTiltVelo(data_type _tiltVelo){ m_tiltCurrentVelocity = _tiltVelo; }

        std::vector<byte> toByte(){
            std::vector<byte> result;
            std::vector<byte> b_panVelo, b_tiltVelo, b_panPos, b_tiltPos;
            b_panPos = Utils::toByte<data_type>(m_panCurrentPosition);
            b_tiltPos = Utils::toByte<data_type>(m_tiltCurrentPosition);
            b_panVelo = Utils::toByte<data_type>(m_panCurrentVelocity);
            b_tiltVelo = Utils::toByte<data_type>(m_tiltCurrentVelocity);
            result.push_back(m_panStabMode);
            result.push_back(m_tiltStabMode);
            result.insert(result.end(), b_panPos.begin(), b_panPos.end());
            result.insert(result.end(), b_tiltPos.begin(), b_tiltPos.end());
            result.insert(result.end(), b_panVelo.begin(), b_panVelo.end());
            result.insert(result.end(), b_tiltVelo.begin(), b_tiltVelo.end());
            return result;
        }

        MotionCStatus* parse(std::vector<byte> _data, index_type _index = 0){
            byte* data = _data.data();
            m_panStabMode =  Utils::toValue<byte>(data,_index);
            m_tiltStabMode = Utils::toValue<byte>(data, _index + sizeof(byte));
            m_panCurrentPosition = Utils::toValue<data_type>(data, _index + sizeof(byte)*2);
            m_tiltCurrentPosition = Utils::toValue<data_type>(data, _index + sizeof(byte)*2 + sizeof(data_type));
            m_panCurrentVelocity = Utils::toValue<data_type>(data, _index + sizeof(byte)*2 + sizeof(data_type)*2);
            m_tiltCurrentVelocity = Utils::toValue<data_type>(data, _index + sizeof(byte)*2 + sizeof(data_type)*3);
            return this;
        }

        MotionCStatus* parse(byte *_data, index_type _index = 0){
            byte* data = _data;
            m_panStabMode =  Utils::toValue<byte>(data,_index);
            m_tiltStabMode = Utils::toValue<byte>(data, _index + sizeof(byte));
            m_panCurrentPosition = Utils::toValue<data_type>(data, _index + sizeof(byte)*2);
            m_tiltCurrentPosition = Utils::toValue<data_type>(data, _index + sizeof(byte)*2 + sizeof(data_type));
            m_panCurrentVelocity = Utils::toValue<data_type>(data, _index + sizeof(byte)*2 + sizeof(data_type)*2);
            m_tiltCurrentVelocity = Utils::toValue<data_type>(data, _index + sizeof(byte)*2 + sizeof(data_type)*3);
            return this;
        }

    private:
        byte m_panStabMode;
        byte m_tiltStabMode;
        data_type m_panCurrentVelocity;
        data_type m_tiltCurrentVelocity;
        data_type m_panCurrentPosition;
        data_type m_tiltCurrentPosition;
    };
}
#endif // MOTIONCSTATUS_H
