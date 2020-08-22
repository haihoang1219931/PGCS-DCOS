#include<iostream>
#include"UavSensorZoom.h"

UavvSensorZoom::UavvSensorZoom(){}

UavvSensorZoom::UavvSensorZoom(unsigned char sensorIndex, short zoomValue)
{
    SensorIndex = sensorIndex;
    ZoomValue = zoomValue;
}

UavvSensorZoom::~UavvSensorZoom(){}

GimbalPacket UavvSensorZoom::Encode()
{
	unsigned char data[6];
    data[0] = SensorIndex;
    data[1] = ZoomFlag;
    ByteManipulation::ToBytes((unsigned short)ZoomValue,Endianness::Big,data,2);
    ByteManipulation::ToBytes((unsigned short)0,Endianness::Big,data,4);
    return GimbalPacket(UavvGimbalProtocol::SensorZoom, data, sizeof(data));
}

ParseResult UavvSensorZoom::TryParse(GimbalPacket packet, UavvSensorZoom *SensorZoom)
{
    if (packet.Data.size() < SensorZoom->Length)
	{
        return ParseResult::InvalidLength;
	}
    unsigned char _Type = packet.Data[0];
    short _ZoomValue = (short)ByteManipulation::ToUInt16(packet.Data.data(),2,Endianness::Big);
	if ((_ZoomValue >= -100) && (_ZoomValue <= 100))
	{
        *SensorZoom = UavvSensorZoom(_Type, _ZoomValue);
        return ParseResult::Success;
	}
    return ParseResult::InvalidData;
}
