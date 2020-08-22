#include "UavvEnableStreamMode.h"

UavvEnableStreamMode::UavvEnableStreamMode(EnableStreamTypeActionFlag type, EnableStreamFrequencyFlag frequency) 
{
	EnableStreamTypeAction = (unsigned char)type;
	EnableStreamFrequency = (unsigned char)frequency;
};
UavvEnableStreamMode::~UavvEnableStreamMode() {}
UavvEnableStreamMode::UavvEnableStreamMode() {}
ParseResult UavvEnableStreamMode::TryParse(GimbalPacket packet, UavvEnableStreamMode *EnableStreamMode)
{
    if (packet.Data.size() < EnableStreamMode->Length)
	{
        return ParseResult::InvalidLength;
	}
    EnableStreamMode->EnableStreamTypeAction = packet.Data[0];
	EnableStreamMode->EnableStreamFrequency = packet.Data[1];
    return ParseResult::Success;
}

GimbalPacket UavvEnableStreamMode::encode()
{
	unsigned char data[2];
	data[0] = (unsigned char)EnableStreamTypeAction;
	data[1] = (unsigned char)EnableStreamFrequency;
	return GimbalPacket(UavvGimbalProtocol::PositionStreaming, data, sizeof(data));
}
