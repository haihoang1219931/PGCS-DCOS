#include "UavvEnableMessageACK.h"

UavvEnableMessageACK::UavvEnableMessageACK() {}
UavvEnableMessageACK::~UavvEnableMessageACK() {}

UavvEnableMessageACK::UavvEnableMessageACK(unsigned char data01,unsigned char data02)
{
    Data01 = data01;
    Data02 = data02;
}

ParseResult UavvEnableMessageACK::TryParse(GimbalPacket packet, UavvEnableMessageACK *Packet)
{
    if (packet.Data.size() < Packet->Length)
	{
        return ParseResult::InvalidLength;
	}
    Packet->Data01 = packet.Data[0];
    Packet->Data02 = packet.Data[1];
    return ParseResult::Success;
}

GimbalPacket UavvEnableMessageACK::Encode()
{
    unsigned char data[2];
    data[0] = Data01;
    data[1] = Data02;
	return GimbalPacket(UavvGimbalProtocol::EnableMessageAcknowledgment, data, sizeof(data));
}
