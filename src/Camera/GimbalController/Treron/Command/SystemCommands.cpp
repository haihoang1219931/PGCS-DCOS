#include "SystemCommands.h"
SystemCommands::SystemCommands(QObject* parent) :
    QObject(parent)
{

}
SystemCommands::~SystemCommands()
{

}

void SystemCommands::setInstallMode(QString mouseMode, QString apMode){
    Eye::KLV packet;
    InstallMode data;
    if(mouseMode == "MOUNT_BELL"){
        data.setMount((byte)Status::InstallMode::MOUNT_BELL);
    }else if(mouseMode == "MOUNT_NOSE"){
        data.setMount((byte)Status::InstallMode::MOUNT_NOSE);
    }
    if(apMode == "AP_GCS"){
        data.setAP(AP_GCS);
    }else if(apMode == "AP_AP"){
        data.setAP(AP_AP);
    }
    packet.key = (key_type) EyePhoenixProtocol::CameraInstall;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();

}

void SystemCommands::getCameraStatus(){
    Eye::KLV packet;
    RequestResponsePacket data;
    data.setPacketID((key_type) EyePhoenixProtocol::RequestResponse);
    packet.key = (key_type) EyePhoenixProtocol::RequestResponse;
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}

void SystemCommands::resetIMU(){
    Eye::KLV packet;
    packet.key = (key_type) EyePhoenixProtocol::IMUReset;
    RapidView data;
    data.setViewMode(0x01);
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}

void SystemCommands::calibIMU(){
    Eye::KLV packet;
    packet.key = (key_type) EyePhoenixProtocol::IMUCalib;
    RapidView data;
    data.setViewMode(0x01);
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
