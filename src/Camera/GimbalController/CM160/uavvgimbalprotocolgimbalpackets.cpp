#include "uavvgimbalprotocolgimbalpackets.h"

UavvGimbalProtocolGimbalPackets::UavvGimbalProtocolGimbalPackets(QObject* parent) :
    QObject(parent)
{

}
void UavvGimbalProtocolGimbalPackets::stowConfiguration(
        unsigned char saveToFlash,
        unsigned char enableStow,
        unsigned short stowTimeout,
        unsigned short stowedOnPan,
        unsigned short stowedOnTilt){
    GimbalPacket payload = UavvStowConfiguration(
                (unsigned char )saveToFlash,
                (unsigned char)enableStow,
                (unsigned short) stowTimeout,
                (unsigned short) stowedOnPan,
                (unsigned short) stowedOnPan).encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::stowConfiguration(){
    GimbalPacket payload = UavvRequestResponse(UavvGimbalProtocol::StowConfiguration).Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::stowMode(QString stowMode){
    UavvStowMode protocol;
    if(stowMode == "ON"){
        protocol.StowMode = (unsigned char)StowModeType::EnterStow;
    }else if(stowMode == "OFF"){
        protocol.StowMode = (unsigned char)StowModeType::ExitStow;
    }
    protocol.Reserverd = (unsigned char) 0;
    GimbalPacket payload = protocol.encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::stowStatusResponse(){
    UavvRequestResponse protocol(UavvGimbalProtocol::SetStowMode);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::initiliseGimbal(bool enable){
    UavvInitialiseGimbal protocol;
    if(enable == true){
        protocol.data01 = (unsigned char)1;
        protocol.data02 = (unsigned char)1;
    }else{
        protocol.data01 = (unsigned char)0;
        protocol.data02 = (unsigned char)0;
    }
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::setPanPosition(float angle){
    UavvSetPanPosition protocol;
    protocol.PanPosition = (float)angle;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::setTiltPosition(float angle){
    UavvSetTiltPosition protocol;
    protocol.TiltPosition = (float)angle;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::setPanTiltPosition(float anglePan, float angleTilt){
    UavvSetPanTiltPosition protocol;
    printf("Set pan/tilt = %.2f/%.2f\r\n",anglePan,angleTilt);
    protocol.PanPosition = (float) anglePan;
    protocol.TiltPosition = (float)angleTilt;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::setPanVelocity(float velocity){
    UavvSetPanVelocity protocol;
    protocol.PanVelocity = (float) velocity;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::setTiltVelocity(float velocity){
    UavvSetTiltVelocity protocol;
    protocol.TiltVelocity = (float) velocity;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::setPanTiltVelocity(float velPan, float velTilt ){
    UavvSetPanTiltVelocity protocol;
    protocol.PanVelocity = velPan;
    protocol.TiltVelocity = velTilt;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::getCurrentGimbalMode(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::TrackingStatus;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::setPrimaryVideo(int data01,int primaryVideoSensor){
    UavvSetPrimaryVideo protocol;
    protocol.Data01 = (unsigned char) 0;
    protocol.PrimaryVideoSensor = (unsigned char)primaryVideoSensor;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::setSceneSteering(bool enable){
    UavvSceneSteering protocol;
    int data01;
    int data02;
    if(enable == true){
        data01= 1;
        data02= 1;
    }else{
        data01= 0;
        data02= 0;
    }
    protocol.SceneSteering = (unsigned char) data01;
    protocol.AutomaticScene = (unsigned char)data02;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::getSceneSteering(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SceneSteering;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::setSceneSteeringConfiguration(SceneSteeringAction autoFlags){
    UavvSceneSteeringConfiguration protocol;
    protocol.sceneSteeringAction = (unsigned char)autoFlags;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolGimbalPackets::getSceneSteeringConfiguration(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SceneSteeringConfig;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
