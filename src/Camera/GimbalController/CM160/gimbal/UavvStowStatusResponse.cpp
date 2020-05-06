#include"UavvStowStatusResponse.h"

UavvStowStatusResponse::UavvStowStatusResponse(StowModeResponse stowmode)
{
    StowMode = (unsigned char)stowmode;
}

UavvStowStatusResponse::~UavvStowStatusResponse() {}
UavvStowStatusResponse::UavvStowStatusResponse() {}
ParseResult UavvStowStatusResponse::TryParse(GimbalPacket packet, UavvStowStatusResponse *StatusResponse)
{
    if (packet.Data.size() < StatusResponse->Length)
	{
        return ParseResult::InvalidLength;
	}
    StatusResponse->StowMode = packet.Data[0];
    StatusResponse->Reserverd = packet.Data[1];
    return ParseResult::Success;
}

GimbalPacket UavvStowStatusResponse::encode()
{
	unsigned char data[2];
    data[0] = StowMode;
    data[1] = Reserverd;
	return GimbalPacket(UavvGimbalProtocol::SetStowMode, data, sizeof(data));
}
