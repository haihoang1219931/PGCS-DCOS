#include "MotionCCommands.h"
MotionCCommands::MotionCCommands(QObject* parent) :
    QObject(parent)
{

}
MotionCCommands::~MotionCCommands()
{

}
void MotionCCommands::changeRapidView(QString mode){
    Eye::KLV packet;
    Eye::RapidView data;

    if(mode == "FRONTVIEW"){
        data.setViewMode((byte)Status::PresetMode::FRONT);
    }else if(mode == "RIGHTWING"){
        data.setViewMode((byte)Status::PresetMode::RIGHT_WING);
    }else if(mode == "NADIR"){
        data.setViewMode((byte)Status::PresetMode::NADIR);
    }else if(mode == "FREE"){
        data.setViewMode((byte)Status::PresetMode::FREE);
    }
    packet.key = (key_type) EyePhoenixProtocol::SetRapidView;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
void MotionCCommands::setPanTiltVelocity(int id, float panVel, float tiltVel){
    Eye::KLV packet;
    PTRateFactor data;
    data.setPTRateFactor((index_type)id, (data_type)panVel,(data_type)tiltVel);
    packet.key = (key_type) EyePhoenixProtocol::SetPanTiltRateFactor;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}

void MotionCCommands::setPanTiltAngle(float panAngle, float tiltAngle){
    Eye::KLV packet;
    PTAngle data;
    data.setPTAngle((data_type)panAngle,(data_type)tiltAngle);
    packet.key = (key_type) EyePhoenixProtocol::SetPanTiltAngle;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}

void MotionCCommands::setPanTiltDiffAngle(float panDif, float tiltDif){
    Eye::KLV packet;
    PTAngleDiff data;
    data.setPTAngleDiff((data_type)panDif,(data_type)tiltDif);
    packet.key = (key_type) EyePhoenixProtocol::SetPanTiltAngleDiff;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
void MotionCCommands::enableGimbalStab(bool stabPan, bool stabTilt){
    Eye::KLV packet;
    GimbalStab data;
    if(stabPan == true){
        data.setStabPan(GIMBAL_PAN_ON);
    }else{
        data.setStabPan(GIMBAL_PAN_OFF);
    }
    if(stabTilt == true){
        data.setStabTilt(GIMBAL_TILT_ON);
    }else{
        data.setStabTilt(GIMBAL_TILT_OFF);
    }
    packet.key = (key_type) EyePhoenixProtocol::GimbalStab;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}

void MotionCCommands::setMCPanParams(float _kp, float _ki){
    Eye::KLV packet;
    MCParams mcParams;
    mcParams.setKi(_ki);
    mcParams.setKp(_kp);
    packet.key = (key_type) EyePhoenixProtocol::MotionPanSetParams;
    packet.data = mcParams.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}

void MotionCCommands::setMCTiltParams(float _kp, float _ki){
    Eye::KLV packet;
    MCParams mcParams;
    mcParams.setKi(_ki);
    mcParams.setKp(_kp);
    packet.key = (key_type) EyePhoenixProtocol::MotionTiltSetParams;
    packet.data = mcParams.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}

