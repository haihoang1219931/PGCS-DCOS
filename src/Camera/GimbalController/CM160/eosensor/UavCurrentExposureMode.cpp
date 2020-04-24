#include<iostream>
#include"UavCurrentExposureMode.h"

UavvCurrentExposureMode::UavvCurrentExposureMode(){}

UavvCurrentExposureMode::UavvCurrentExposureMode(unsigned char index, ExposureMode mode)
{
    Mode = mode;
    Index = index;
}

UavvCurrentExposureMode::~UavvCurrentExposureMode(){}

GimbalPacket UavvCurrentExposureMode::Encode()
{
	unsigned char data[2];
    data[0] = (unsigned char)Index;
    data[1] = (unsigned char)Mode;
    return GimbalPacket(UavvGimbalProtocol::EoExposureMode, data, sizeof(data));
}

ParseResult UavvCurrentExposureMode::TryParse(GimbalPacket packet, UavvCurrentExposureMode *CurrentExposureMode)
{
    if (packet.Data.size() < CurrentExposureMode->Length)
	{
        return ParseResult::InvalidLength;
	}
	unsigned char index = packet.Data[0];
	ExposureMode mode;
	switch (packet.Data[1])
	{
	case 0x00:
        mode = ExposureMode::Automatic;
		break;
	case 0x01:
        mode = ExposureMode::IrisPriority;
		break;
	case 0x02:
        mode = ExposureMode::ShutterPriority;
		break;
	case 0x03:
        mode = ExposureMode::Manual;
		break;
	default:
        return ParseResult::InvalidData;
	}
    *CurrentExposureMode = UavvCurrentExposureMode(index, mode);
    return ParseResult::Success;
};
