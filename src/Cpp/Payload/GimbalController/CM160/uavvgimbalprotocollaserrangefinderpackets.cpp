#include "uavvgimbalprotocollaserrangefinderpackets.h"

UavvGimbalProtocolLaserRangeFinderPackets::UavvGimbalProtocolLaserRangeFinderPackets(QObject* parent) :
    QObject(parent)
{

}
void UavvGimbalProtocolLaserRangeFinderPackets::getDistance(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::LaserRange;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolLaserRangeFinderPackets::laserRangeStart(bool start){
    LaserRangeStart protocol;
    protocol.Mesuaring = start == true?(unsigned char)1:(unsigned char)0;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolLaserRangeFinderPackets::getLaserDeviceStatus(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::LaserDeviceStatus;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolLaserRangeFinderPackets::armLaserDevice(bool arm){
    ArmLaserDevice protocol;
    protocol.Arm = arm == true?(unsigned char)1:(unsigned char)0;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolLaserRangeFinderPackets::fireLaserDevice(bool fire){
    FireLaserDevice protocol;
    protocol.Fire = fire == true?(unsigned char)3:(unsigned char)0;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
