#include "uavvgimbalprotocolgeopointingpackets.h"

UavvGimbalProtocolGeoPointingPackets::UavvGimbalProtocolGeoPointingPackets(QObject* parent) :
    QObject(parent)
{

}
void UavvGimbalProtocolGeoPointingPackets::setGeolockLocation(
        QString actionFlag,
        float latitude,
        float longtitude,
        float height){
    UavvSetGeolockLocation protocol;
    //GeoLockActionFlag ;
    if(actionFlag == "Disable Geolock"){
        protocol.Flag=GeoLockActionFlag::DisableGeoLock;
    }else if(actionFlag == "Enable Geolock at Cross hair"){
        protocol.Flag=GeoLockActionFlag::EnableGeoLockAtCrossHair;
    }else if(actionFlag == "Enable Geolock at Coordinate"){
        protocol.Flag=GeoLockActionFlag::EnableGeoLockAtCoordinateGimbal;
    }

    protocol.Longitude=longtitude;
    protocol.Latitude=latitude;
    protocol.Height=height;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::currentGeolockSetpoint(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::CurrentGeolockSetpoint;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::currentTargetLocation(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::CurrentTargetLocation;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::currentCornerLocation(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::CurrentCornerLocations;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::terrainHeight(){

}
void UavvGimbalProtocolGeoPointingPackets::seedTerrainHeight(float latitude,
                                   float longtitude,
                                   float height){
    UavvSeedTerrainHeight protocol;
    protocol.Longtitude=longtitude;
    protocol.Latitude=latitude;
    protocol.Height=height;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::seedTerrainHeight(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::SeedTerrainHeight;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::gnssStatus(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::ImuStatus;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::platformOrientation(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::PlatformOrientation;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::platformPosition(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::PlatformPosition;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::gimbalOrientationOffset(
        int reserve,
        int rollOffset,
        int pitchOffset,
        int yawOffset){
    UavvGimbalOrientationOffset protocol;
    protocol.Roll = (short)rollOffset;
    protocol.Pitch = (short)pitchOffset;
    protocol.Yaw = (short)yawOffset;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::gimbalOrientationOffset(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::PlatformOrientationOffset;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::altitudeOffset(unsigned char reserve,
                                short altitude){
    UavvAltitudeOffset protocol;
    protocol.Reserved = reserve;
    protocol.Altitude = altitude;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::altitudeOffset(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::AltitudeOffset;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::gimbalMisalignmentOffset(
      QString mountType,
      float panMisalign,
      float tiltMisalign){
    UavvGimbalMisalignmentOffset protocol;
    if(mountType == "CradleDown"){
        protocol.MountType = (unsigned char)0;
    }else if(mountType == "NotUsed"){
        protocol.MountType = (unsigned char)1;
    }else if(mountType == "CradleUp"){
        protocol.MountType = (unsigned char)2;
    }else if(mountType == "Nose"){
        protocol.MountType = (unsigned char)3;
    }
    protocol.Pan = (short)(panMisalign/360*32768);
    protocol.Tilt = (short)(tiltMisalign/360*32768);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::gimbalMisalignmentOffset(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::IMUTranslationOffset;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::sendExternalAltitude(float roll,float pitch, float yaw){
    UavvExternalAltitude protocol;
    protocol.Roll = roll;
    protocol.Pitch = pitch;
    protocol.Yaw = yaw;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}

void UavvGimbalProtocolGeoPointingPackets::sendExternalPosition(float lat,float lon, float alt){
    UavvExternalPosition protocol;
    protocol.Latitude = lat;
    protocol.Longtitude = lon;
    protocol.Altitude = alt;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGeoPointingPackets::sendExternalElevation(float lat,float lon, float alt){
    UavvSeedTerrainHeight protocol;
    protocol.Latitude = lat;
    protocol.Longtitude = lon;
    protocol.Height = alt;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
