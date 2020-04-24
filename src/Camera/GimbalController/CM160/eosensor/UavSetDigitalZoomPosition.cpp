#include<iostream>
#include"UavSetDigitalZoomPosition.h"

UavvSetDigitalZoomPosition::UavvSetDigitalZoomPosition(){}

UavvSetDigitalZoomPosition::UavvSetDigitalZoomPosition(unsigned short position)
{
    Position = position;
}

UavvSetDigitalZoomPosition::~UavvSetDigitalZoomPosition(){}

GimbalPacket UavvSetDigitalZoomPosition::Encode()
{
	unsigned char data[2];
    ByteManipulation::ToBytes(Position,Endianness::Big,data,0);
    return GimbalPacket(UavvGimbalProtocol::SetCameraDigitalZoomPosition, data, sizeof(data));
}

ParseResult UavvSetDigitalZoomPosition::TryParse(GimbalPacket packet, UavvSetDigitalZoomPosition *SetDigitalZoomPosition)
{
    if (packet.Data.size() < SetDigitalZoomPosition->Length)
	{
        return ParseResult::InvalidLength;
	}
    unsigned short digitalZoomPosition =ByteManipulation::ToUInt16(packet.Data.data(),0,Endianness::Big);
	if (digitalZoomPosition>0x4000)
        return ParseResult::InvalidData;
	*SetDigitalZoomPosition = UavvSetDigitalZoomPosition(digitalZoomPosition);
    return ParseResult::Success;
}
