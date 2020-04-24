#include "uavvgimbalprotocolirsensorpackets.h"

UavvGimbalProtocolIRSensorPackets::UavvGimbalProtocolIRSensorPackets(QObject* parent) :
    QObject(parent)
{

}
void UavvGimbalProtocolIRSensorPackets::setIRSensorTempResponse(){
    UavvIRSensorTemperatureResponse protocol;
    protocol.Temperature = 0;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRSensorTempResponse(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::IRSensorTemperature;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRVideoModulation(
        unsigned char reverse,VideoModulation moduleFlag){
    UavvSetIRVideoModulation protocol;
    protocol.Reserved = reverse;
    protocol.VideoModule = moduleFlag;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRVideoModulation(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRVideoStandard;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRZoom(
        int zoomFlag){
    UavvSetIRZoom protocol;
    protocol.Flag = (ZoomFlag)zoomFlag;
    protocol.Reserved = 0;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRAGC(
        AGCMode agcMode){
    UavvSetIRAGCMode protocol;
    protocol.Mode = agcMode;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRBrightness(
        unsigned short brightness){
    UavvSetIRBrightness protocol;
    protocol.Brightness = brightness;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRBrightness(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRBrightness;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::enableIRIsotherm(
        bool enable){
    UavvEnableIRIsotherm protocol;
    protocol.Status = enable == true?IsothermStatus::Enable:IsothermStatus::Disable;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::enableIRIsotherm(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::EnableIsotherm;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::resetIRCamera(
        unsigned char data01,unsigned char data02){
    UavvResetIRCamera protocol;
    protocol.Data01 = data01;
    protocol.Data02 = data02;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setFFCMode(
        bool ffcMode){
    UavvSetFFCMode protocol;
    protocol.Mode = (FFCMode)ffcMode;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::performFlatFieldCorrection(
        unsigned char reserve,unsigned char FFC){
    UavvPerformFFC protocol;
    protocol.PerformFFC = FFC;
    protocol.Reserved = reserve;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setFFCTempDelta(unsigned short temp){
    UavvSetFFCTemperatureDelta protocol;
    protocol.TemperatureDelta = temp;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setMWIRTempPreset(
        QString preset){
    UavvMWIRTempPreset protocol;
    qDebug("set MWIRTempPreset to %s",preset.toStdString().c_str());
    if(preset.toStdString()== "HOT"){
        protocol.Mode[0] = 0x01;
        protocol.Mode[1] = 0x01;
    }else if(preset.toStdString()== "MEDIUM"){
        protocol.Mode[0] = 0x01;
        protocol.Mode[1] = 0x02;
    }else if(preset.toStdString()== "COLD"){
        protocol.Mode[0] = 0x01;
        protocol.Mode[1] = 0x03;
    }
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRPalette(
        QString palette){
    UavvSetIRPalette protocol;
    qDebug("set palette to %s",palette.toStdString().c_str());
    if(palette.toStdString()== "Whitehot"){
        protocol.Mode = PaletteMode::Whitehot;
    }else if(palette.toStdString()== "Blackhot"){
        protocol.Mode = PaletteMode::Blackhot;
    }else if(palette.toStdString()== "Fusion"){
        protocol.Mode = PaletteMode::Fusion;
    }else if(palette.toStdString()== "Rain"){
        protocol.Mode = PaletteMode::Rain;
    }
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRPalette(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRPalette;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRVideoOrientation(
        unsigned char reserve,VideoOrientationMode orienOption){
    UavvSetIRVideoOrientation protocol;
    protocol.Mode = orienOption;
    protocol.Reserved = reserve;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRVideoOrientation(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRVideoOrientation;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRContrast(
        unsigned char data01, unsigned char value){
    UavvSetIRContrast protocol;
    protocol.Contrast = value;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRContrast(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRContrast;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRBrightnessBias(short value){
    UavvSetIRBrightnessBias protocol;
    protocol.Bias = value;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRBrightnessBias(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRBrightnessBias;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}

void UavvGimbalProtocolIRSensorPackets::setIRPlateanLevel(unsigned short value){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRPlateauLevel;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRPlateanLevel(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRPlateauLevel;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRImageTransformTableMidpoint(
        unsigned char data01,unsigned char value){
    UavvSetIRITTMidpoint protocol;
    protocol.Data01 = data01;
    protocol.Midpoint = value;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRImageTransformTableMidpoint(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRITTMid;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRMaxGain(unsigned short value){
    UavvSetIRMaxGain protocol;
    protocol.MaxGain = value;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRMaxGain(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRMAXAGC;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRGainMode(
        unsigned char data01,GainMode mode){
    UavvSetIRGainMode protocol;
    protocol.Mode = mode;
    protocol.Data01 = data01;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRGainMode(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRGainMode;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::setIRDDE(
        ManualDDEStatus DDE,unsigned char sharpness){
    UavvSetDynamicDDE protocol;
    protocol.Status = DDE;
    protocol.Sharpness = sharpness;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolIRSensorPackets::getIRDDE(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetIRDDE;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
