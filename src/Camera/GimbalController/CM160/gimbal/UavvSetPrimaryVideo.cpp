#include "UavvSetPrimaryVideo.h"

UavvSetPrimaryVideo::UavvSetPrimaryVideo() {}
UavvSetPrimaryVideo::~UavvSetPrimaryVideo() {}

UavvSetPrimaryVideo::UavvSetPrimaryVideo(PrimaryVideoSensorType setprimaryvideo)
{
    PrimaryVideoSensor = (unsigned char)setprimaryvideo;
}

ParseResult UavvSetPrimaryVideo::TryParse(GimbalPacket packet, UavvSetPrimaryVideo *setprimary)
{
    if (packet.Data.size() < setprimary->Length)
	{
        return ParseResult::InvalidLength;
	}
    setprimary->PrimaryVideoSensor = packet.Data[1];
    return ParseResult::Success;
}

GimbalPacket UavvSetPrimaryVideo::Encode()
{
	unsigned char data[2];
    data[0] = Data01;
    data[1] = PrimaryVideoSensor;
	GimbalPacket result = GimbalPacket(UavvGimbalProtocol::ToggleVideoOutput, data, sizeof(data));
	return result;
}
