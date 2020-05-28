#include<iostream>
#include"UavPerformFFC.h"

UavvPerformFFC::UavvPerformFFC(){}

UavvPerformFFC::~UavvPerformFFC(){}

GimbalPacket UavvPerformFFC::Encode()
{
	unsigned char data[2];
    data[0] = Reserved;
    data[1] = PerformFFC;
    return GimbalPacket(UavvGimbalProtocol::PerformFFC, data, sizeof(data));
}

ParseResult UavvPerformFFC::TryParse(GimbalPacket packet, UavvPerformFFC *performFFC)
{
    if (packet.Data.size() < performFFC->Length)
	{
        return ParseResult::InvalidLength;
	}

	if ((packet.Data[0] != 0x00) || (packet.Data[1] != 0x01))
        return ParseResult::InvalidData;
    performFFC->Reserved = packet.Data[0];
    performFFC->PerformFFC = packet.Data[1];
    return ParseResult::Success;
};
