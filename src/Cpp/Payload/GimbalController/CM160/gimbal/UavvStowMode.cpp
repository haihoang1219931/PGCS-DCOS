#include "UavvStowMode.h"

UavvStowMode::UavvStowMode(StowModeType stowmode)
{
    StowMode = (unsigned char)stowmode;
}
UavvStowMode::UavvStowMode() {}
UavvStowMode::~UavvStowMode() {}

ParseResult UavvStowMode::TryParse(GimbalPacket packet, UavvStowMode *stowmode)
{
    if (packet.Data.size()< stowmode->Length)
	{
        return ParseResult::InvalidLength;
	}
    stowmode->StowMode = packet.Data[0];
    stowmode->Reserverd = packet.Data[1];
    return ParseResult::Success;
}

GimbalPacket UavvStowMode::encode()
{
	unsigned char data[2];
    data[0] = StowMode;
    data[1] = Reserverd;
	return GimbalPacket(UavvGimbalProtocol::SetStowMode, data, sizeof(data));
}
