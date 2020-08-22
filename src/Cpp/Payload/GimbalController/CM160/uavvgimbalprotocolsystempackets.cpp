#include "uavvgimbalprotocolsystempackets.h"

UavvGimbalProtocolSystemPackets::UavvGimbalProtocolSystemPackets(QObject* parent) :
    QObject(parent)
{

}
void UavvGimbalProtocolSystemPackets::enableGyroStabilisation(bool enable){
    GimbalPacket payload;
    vector<unsigned char> packet;
    switch (enable) {
    case true:
        payload = UavvEnableGyroStabilisation(1,1).Encode();
        packet = payload.encode();
        _udpSocket->write((const char*)packet.data(),packet.size());
        break;
    case false:
        payload = UavvEnableGyroStabilisation(0,0).Encode();
        packet = payload.encode();
        _udpSocket->write((const char*)packet.data(),packet.size());
        break;
    }
}

void UavvGimbalProtocolSystemPackets::getGyroStabilisation(){
    GimbalPacket payload = UavvRequestResponse(UavvGimbalProtocol::GyroStablisation).Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}

void UavvGimbalProtocolSystemPackets::messageAcknolegdeResponse(int packetID){
    GimbalPacket payload = UavvMessageACKResponse((unsigned char)packetID).Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolSystemPackets::enableMessageAcknowledge(int data1, int data2){
    GimbalPacket payload = UavvEnableMessageACK((unsigned char)data1,(unsigned char)data2).Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolSystemPackets::enableStreamingMode(EnableStreamTypeActionFlag mode,
                                                          EnableStreamFrequencyFlag freq){
    GimbalPacket payload = UavvEnableStreamMode(mode,freq).encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolSystemPackets::getVersion(){
    GimbalPacket payload = UavvRequestResponse(UavvGimbalProtocol::Version).Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolSystemPackets::setProtocolVersion(){}
void UavvGimbalProtocolSystemPackets::getProtocolVersion(){}
void UavvGimbalProtocolSystemPackets::getGimbalSerialNumberResponse(){

}
void UavvGimbalProtocolSystemPackets::requestResponse(int packetID){
    GimbalPacket payload = UavvRequestResponse(packetID).Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolSystemPackets::configurePacketRates(vector<PacketRate> lstPacketRate){
    GimbalPacket payload = UavvConfigurePacketRates(true,lstPacketRate).Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolSystemPackets::setSystemTime(int second){
    time_t t = time(0);
    int localHour,gmtHour;
    struct tm * lcl = localtime( & t );
    localHour = lcl->tm_hour;
    char timestamp[64];
    sprintf(timestamp, "Local: %04d-%02d-%02d %02d:%02d:%02d",
    (lcl->tm_year + 1900),
    lcl->tm_mon + 1,
    lcl->tm_mday,
    lcl->tm_hour,
    lcl->tm_min,
    lcl->tm_sec);
    printf("%s equal %d seconds\r\n",timestamp,second);
    struct tm * gmt = gmtime( & t );
    gmtHour = gmt->tm_hour;
    char gmtTimestamp[64];
    sprintf(gmtTimestamp, "GMT: %04d-%02d-%02d %02d:%02d:%02d",
    (gmt->tm_year + 1900),
    gmt->tm_mon + 1,
    gmt->tm_mday,
    gmt->tm_hour,
    gmt->tm_min,
    gmt->tm_sec);
    printf("%s equal %d seconds\r\n",gmtTimestamp,second);
    printf("Different time: [%d - %d] %dH to %dS\r\n",
           localHour,gmtHour,
           localHour - gmtHour,
           (localHour - gmtHour)*3600);
    UavvSetSystemTime newSystemTime;
    newSystemTime.second = t + (localHour-gmtHour)*3600;
    GimbalPacket payload = newSystemTime.encode();
    vector<unsigned char> packet = payload.encode();
    printf("Packet: ");
    for(int i=0; i < packet.size(); i++){
        printf("%02X ",packet[i]);
    }
    printf("\r\n");
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolSystemPackets::setNetworkConfiguration(
        uint8_t reserved,
        uint8_t ipType,
        uint32_t ipAddress,
        uint32_t subnetMask,
        uint32_t gateWay,
        uint16_t reserved01,
        uint16_t reserved02,
        uint8_t reserved03,
        uint8_t saved){
    UavvNetworkConfiguration newNetwork;
    GimbalPacket payload = newNetwork.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolSystemPackets::getNetworkConfiguration(){
    GimbalPacket payload = UavvRequestResponse(UavvGimbalProtocol::NetworkConfiguration).Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolSystemPackets::saveParameters(uint8_t pa01,uint8_t saveParametersFlag){
    GimbalPacket payload = UavvSaveParameters((unsigned char)saveParametersFlag).Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
