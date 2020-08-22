#include"UavSetIris.h"

UavvSetIris::UavvSetIris(){}

UavvSetIris::UavvSetIris(unsigned char _FStopIndex)
{
    FStopIndex = _FStopIndex;
}

UavvSetIris::~UavvSetIris(){}

GimbalPacket UavvSetIris::Encode()
{
	unsigned char data[2];
    data[0] = Reserved;
    data[1] = FStopIndex;
    return GimbalPacket(UavvGimbalProtocol::SetIris, data, sizeof(data));
}

ParseResult UavvSetIris::TryParse(GimbalPacket packet, UavvSetIris *SetIris)
{
    if (packet.Data.size() < SetIris->Length)
	{
        return ParseResult::InvalidLength;
	}
    SetIris->Reserved = packet.Data[0];
    if (packet.Data[1]<0x16)
	{

        SetIris->FStopIndex = packet.Data[1];
        return ParseResult::Success;
	}
    return ParseResult::InvalidData;
}
