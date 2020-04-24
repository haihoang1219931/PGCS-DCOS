#include<iostream>
#include"UavGetZoomPosition.h"

UavvGetZoomPosition::UavvGetZoomPosition(){}

UavvGetZoomPosition::~UavvGetZoomPosition(){}

GimbalPacket UavvGetZoomPosition::Encode()
{
	unsigned char data[2];
    data[0] = Data01;
    data[1] = Data02;
    return GimbalPacket(UavvGimbalProtocol::QueryZoomPosition, data, sizeof(data));
}

ParseResult UavvGetZoomPosition::TryParse(GimbalPacket packet, UavvGetZoomPosition *GetZoomPosition)
{
    if (packet.Data.size() < GetZoomPosition->Length)
	{
        return ParseResult::InvalidLength;
	}
	if ((packet.Data[0] == 0x00) && (packet.Data[1] == 0x00))
	{
        *GetZoomPosition = UavvGetZoomPosition();
        return ParseResult::Success;
	}
	else
        return ParseResult::InvalidData;
	
}
