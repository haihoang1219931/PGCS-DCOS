#include<iostream>
#include"UavInvertPicture.h"

UavvInvertPicture::UavvInvertPicture(){}
UavvInvertPicture::UavvInvertPicture(bool invert){
    if (invert) {
        Mode = InvertMode::Invert;
    }else{
        Mode = InvertMode::Normal;
    }
}

UavvInvertPicture::UavvInvertPicture(InvertMode mode)
{
    Mode = mode;
}

UavvInvertPicture::~UavvInvertPicture(){}

GimbalPacket UavvInvertPicture::Encode()
{
	unsigned char data[2];
    if (Mode == InvertMode::Normal)
	{
		data[0] = 0;
		data[1] = 0;
	}
	else
	{
		data[0] = 1;
		data[1] = 1;
	}
    return GimbalPacket(UavvGimbalProtocol::SetEOSensorVideoMode, data, sizeof(data));
}

ParseResult UavvInvertPicture::TryParse(GimbalPacket packet, UavvInvertPicture *InvertPicture)
{
    if (packet.Data.size() < InvertPicture->Length)
	{
        return ParseResult::InvalidLength;
	}
	InvertMode mode;
	if ((packet.Data[0] == 0x00) && (packet.Data[1] == 0x00))
	{
        mode = InvertMode::Normal;
	}
	else if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x01))
	{
        mode = InvertMode::Invert;
	}
	else
        return ParseResult::InvalidData;
    *InvertPicture = UavvInvertPicture(mode);
    return ParseResult::Success;
};
