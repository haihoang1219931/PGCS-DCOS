#include "uavvgimbalprotocoleosensorpackets.h"

UavvGimbalProtocolEOSensorPackets::UavvGimbalProtocolEOSensorPackets(QObject* parent) :
    QObject(parent)
{

}
void UavvGimbalProtocolEOSensorPackets::enableEOSensor(bool enable){
    UavvEnableEOSensor protocol(enable);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::getEnableEOSensor(){
    UavvRequestResponse protocol(UavvGimbalProtocol::SensorEnable);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::enableDigitalZoom(bool enable){
    UavvEnableDigitalZoom protocol(enable);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::combinedZoomEnable(bool enable){
    UavvCombinedZoomEnable protocol(enable);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::setDigitalZoomPosition(int position){
    UavvSetDigitalZoomPosition protocol;
    protocol.Position = (unsigned short)position;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::setDigitalZoomVelocity(QString status,
                                                               int velocity){
    UavvSetDigitalZoomVelocity protocol;
    protocol.Velocity = (unsigned char)velocity;
    //(status,velocity);
    if(status == "Zoomin"){
        protocol.Status = ZoomStatus::Zoomin;
    }else if(status == "Zoomout"){
        protocol.Status = ZoomStatus::Zoomout;
    }else{
        protocol.Status = ZoomStatus::Stop;
    }
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::setEOOpticalZoomVelocity(QString status,
                                                                 int velocity){
    UavvSetEOOpticalZoomVelocity protocol;
    //(status,velocity);
    protocol.Velocity = (unsigned short)velocity;
    //(status,velocity);
    if(status == "ZOOM_IN"){
        protocol.Status = ZoomStatus::Zoomin;
    }else if(status == "ZOOM_OUT"){
        protocol.Status = ZoomStatus::Zoomout;
    }else if(status == "ZOOM_STOP"){
        protocol.Status = ZoomStatus::Stop;
    }
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::setEOOpticalZoomPosition(int zoomPosition){
    printf("%s %d\r\n",__func__,zoomPosition);
    UavvSetEOOpticalZoomPosition protocol;
    switch(zoomPosition){
        case 1:
            protocol.Position = (unsigned short)0x0000;
            break;
        case 2:
            protocol.Position = (unsigned short)0x7223;
            break;
        case 3:
            protocol.Position = (unsigned short)0x9132;
            break;
        case 4:
            protocol.Position = (unsigned short)0x833b;
            break;
        case 5:
            protocol.Position = (unsigned short)0xB041;
            break;
        case 6:
            protocol.Position = (unsigned short)0x6846;
            break;
        case 7:
            protocol.Position = (unsigned short)0xFB49;
            break;
        case 8:
            protocol.Position = (unsigned short)0x3C4D;
            break;
        case 9:
            protocol.Position = (unsigned short)0x0050;
            break;
        case 10:
            protocol.Position = (unsigned short)0x7052;
            break;
        case 11:
            protocol.Position = (unsigned short)0x8D54;
            break;
        case 12:
            protocol.Position = (unsigned short)0xAA56;
            break;
        case 13:
            protocol.Position = (unsigned short)0x9E58;
            break;
        case 14:
            protocol.Position = (unsigned short)0x685A;
            break;
        case 15:
            protocol.Position = (unsigned short)0xD35B;
            break;
        case 16:
            protocol.Position = (unsigned short)0x2B5D;
            break;
        case 17:
            protocol.Position = (unsigned short)0x4F5E;
            break;
        case 18:
            protocol.Position = (unsigned short)0x485F;
            break;
        case 19:
            protocol.Position = (unsigned short)0x1860;
            break;
        case 20:
            protocol.Position = (unsigned short)0xBF60;
            break;
        case 21:
            protocol.Position = (unsigned short)0x6561;
            break;
        case 22:
            protocol.Position = (unsigned short)0xE261;
            break;
        case 23:
            protocol.Position = (unsigned short)0x5F62;
            break;
        case 24:
            protocol.Position = (unsigned short)0xB262;
            break;
        case 25:
            protocol.Position = (unsigned short)0x0663;
            break;
        case 26:
            protocol.Position = (unsigned short)0x5963;
            break;
        case 27:
            protocol.Position = (unsigned short)0x8363;
            break;
        case 28:
            protocol.Position = (unsigned short)0xAC63;
            break;
        case 29:
            protocol.Position = (unsigned short)0xD663;
            break;
        case 30:
            protocol.Position = (unsigned short)0x0064;
            break;
    }
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::enableAutoFocus(bool enable){
    UavvEnableAutoFocus protocol(enable);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::disableInfraredCutFilter(bool disable){
    UavvDisableInfraredCutFilter protocol(disable);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::setDefog(QString flag){
    UavvDefog protocol;
    if(flag == "OFF"){
        protocol.Flag = FlagFog::FDisable;
        protocol.Strength = StrengthFog::SDisable;
    }else if(flag == "AUTO"){
        protocol.Flag = FlagFog::EnableAuto;
    }else if(flag == "LOW"){
        protocol.Flag = FlagFog::EnableManual;
        protocol.Strength = StrengthFog::Low;
    }else if(flag == "MEDIUM"){
        protocol.Flag = FlagFog::EnableManual;
        protocol.Strength = StrengthFog::Medium;
    }else if(flag == "HIGH"){
        protocol.Flag = FlagFog::EnableManual;
        protocol.Strength = StrengthFog::High;
    }
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::getDefog(){
    UavvRequestResponse protocol(UavvGimbalProtocol::SensorDefog);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::setFocus(int focus){
    UavvSetFocus protocol;
    protocol.FocusPosition = (unsigned short)focus;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::enableManualIris(bool enable){
    UavvEnableManualIris protocol(enable);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::setIris(int  data01, int data02){
    UavvSetIris protocol;
    protocol.Reserved = (unsigned char)data01;
    protocol.FStopIndex = (unsigned char)data02;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::enableLensStabilisation(bool enable){
    UavvEnableLensStabilization protocol(enable);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::getCurrentExpousureMode(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::EoExposureMode;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::invertPicture(bool invert){
    UavvInvertPicture protocol(invert);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::setShutterSpeed(int data01, int data02){
    UavvSetShutterSpeed protocol;
    protocol.Reserved = (unsigned char)data01;
    protocol.ShutterSpeed = (unsigned char)data02;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::getShutterSpeed(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetShutterSpeed;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::setCameraGain(int data01, int data02){

}
void UavvGimbalProtocolEOSensorPackets::getCameraGain(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::SetCameraGain;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::setEOSensorVideoMode(int reverse, int HDModeFlag){
    UavvSetEOSensorVideoMode protocol;
    protocol.Reserved = (unsigned char)reverse;
    protocol.VideoMode = (unsigned char)HDModeFlag;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::enableManualShutterMode(bool enable){
    UavvEnableManualShutterMode protocol(enable);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::enableAutoExposure(bool enable){
    UavvEnableAutoExposure protocol(enable);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::getZoomPosition(){
    UavvGetZoomPosition protocol;
    protocol.Data01 = 0;
    protocol.Data02 = 0;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::setSensorZoom(int sensorIndex,int zoomFlag,
                   int zoomValue, int reverse){
    UavvSensorZoom protocol((unsigned char)zoomFlag,(short)zoomValue);
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolEOSensorPackets::getSensorsCurrentFOV(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char)
            UavvGimbalProtocol::SensorFieldOfView;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
