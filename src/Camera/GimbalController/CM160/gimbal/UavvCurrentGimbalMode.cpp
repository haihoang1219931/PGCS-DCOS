#include"UavvCurrentGimbalMode.h"

UavvCurrentGimbalMode::UavvCurrentGimbalMode() {}
UavvCurrentGimbalMode::~UavvCurrentGimbalMode() {}
UavvCurrentGimbalMode::UavvCurrentGimbalMode(CurrentGimbalMode gimbalmode)
{
    GimbalMode = (unsigned char)gimbalmode;
}

ParseResult UavvCurrentGimbalMode::TryParse(GimbalPacket packet, UavvCurrentGimbalMode *gimmode)
{
    if (packet.Data.size() < gimmode->Length)
	{
        return ParseResult::InvalidLength;
	}
    gimmode->Reserved = packet.Data[0];
    gimmode->GimbalMode = packet.Data[1];
    return ParseResult::Success;
}

GimbalPacket UavvCurrentGimbalMode::encode()
{
	unsigned char data[2];
    data[0] = Reserved;
    data[1] = GimbalMode;
	GimbalPacket result = GimbalPacket(UavvGimbalProtocol::TrackingStatus, data, sizeof(data));
	return result;
}
