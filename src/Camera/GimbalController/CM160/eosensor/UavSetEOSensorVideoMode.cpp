#include<iostream>
#include"UavSetEOSensorVideoMode.h"

UavvSetEOSensorVideoMode::UavvSetEOSensorVideoMode(){}

UavvSetEOSensorVideoMode::UavvSetEOSensorVideoMode(unsigned char reserved, unsigned char videoMode)
{
    Reserved = reserved;
    VideoMode = videoMode;
}

UavvSetEOSensorVideoMode::~UavvSetEOSensorVideoMode(){}

GimbalPacket UavvSetEOSensorVideoMode::Encode()
{
	unsigned char data[2];
    data[0] = Reserved;
    data[1] = VideoMode;
    return GimbalPacket(UavvGimbalProtocol::SetEOSensorVideoMode, data, sizeof(data));
}

ParseResult UavvSetEOSensorVideoMode::TryParse(GimbalPacket packet, UavvSetEOSensorVideoMode *SetEOSensorVideoMode)
{
    if (packet.Data.size() < SetEOSensorVideoMode->Length)
	{
        return ParseResult::InvalidLength;
	}
	unsigned char data0 = packet.Data[0];
	unsigned char mode;
	if (packet.Data[1] < 0x13)
	{
		mode = packet.Data[1];
	}
	else
        return ParseResult::InvalidData;
    *SetEOSensorVideoMode = UavvSetEOSensorVideoMode(data0, mode);
    return ParseResult::Success;
};
