#include<iostream>
#include"UavSetIRContrast.h"

UavvSetIRContrast::UavvSetIRContrast(){}

UavvSetIRContrast::UavvSetIRContrast(unsigned char contrast)
{
	setContrast(contrast);
}

UavvSetIRContrast::~UavvSetIRContrast(){}

GimbalPacket UavvSetIRContrast::Encode()
{
	unsigned char data[2];
	data[0] = 0x01;
	data[1] = getContrast();
    return GimbalPacket(UavvGimbalProtocol::SetIRContrast, data, sizeof(data));
}

ParseResult UavvSetIRContrast::TryParse(GimbalPacket packet, UavvSetIRContrast *SetIRContrast)
{
    if (packet.Data.size() < SetIRContrast->Length)
	{
        return ParseResult::InvalidLength;
	}

	
	if (packet.Data[0] != 0x01)
        return ParseResult::InvalidData;
	unsigned char _contrast = packet.Data[1];
	if (_contrast>128)
        return ParseResult::InvalidData;

    *SetIRContrast = UavvSetIRContrast(_contrast);
    return ParseResult::Success;
};
