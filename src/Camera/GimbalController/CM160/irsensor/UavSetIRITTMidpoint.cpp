#include<iostream>
#include"UavSetIRITTMidpoint.h"

UavvSetIRITTMidpoint::UavvSetIRITTMidpoint()
{
	setMidpoint(127);
}

UavvSetIRITTMidpoint::UavvSetIRITTMidpoint(unsigned char midpoint)
{
	setMidpoint(midpoint);
}

UavvSetIRITTMidpoint::~UavvSetIRITTMidpoint(){}

GimbalPacket UavvSetIRITTMidpoint::Encode()
{
	unsigned char data[2];
    data[0] = Data01;
	data[1] = getMidpoint();
    return GimbalPacket(UavvGimbalProtocol::SetIRITTMid, data, sizeof(data));
}

ParseResult UavvSetIRITTMidpoint::TryParse(GimbalPacket packet, UavvSetIRITTMidpoint *SetIRITTMidpoint)
{
    if (packet.Data.size() < SetIRITTMidpoint->Length)
	{
        return ParseResult::InvalidLength;
	}


	if (packet.Data[0] != 0x01)
        return ParseResult::InvalidData;
	unsigned char _midpoint = packet.Data[1];
    *SetIRITTMidpoint = UavvSetIRITTMidpoint(_midpoint);
    return ParseResult::Success;
}
