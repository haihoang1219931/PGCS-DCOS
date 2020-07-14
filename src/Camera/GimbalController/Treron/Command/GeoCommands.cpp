#include "GeoCommands.h"
GeoCommands::GeoCommands(QObject* parent) :
    QObject(parent)
{

}
GeoCommands::~GeoCommands()
{

}

void GeoCommands::sendGPSData(double pn, double pe, double pd){
    Eye::KLV packet;
    GPSData data((data_type) pn, (data_type) pe, (data_type) pd);
    packet.key = (key_type) EyePhoenixProtocol::GPSData;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}

void GeoCommands::sendGeoLocation(){
    Eye::KLV packet;
    EnGeoLocation data;
//    printf("\nsend GeoLocation Command %s", data.getEnGeoLocation()?"true":"false");
    packet.key = (key_type) EyePhoenixProtocol::GeoLocation;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
