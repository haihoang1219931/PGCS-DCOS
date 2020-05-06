#include<iostream>
#include"UavEnableIRIsotherm.h"

UavvEnableIRIsotherm::UavvEnableIRIsotherm(){}

UavvEnableIRIsotherm::UavvEnableIRIsotherm(IsothermStatus status)
{
    Status = status;
}

UavvEnableIRIsotherm::~UavvEnableIRIsotherm(){}

GimbalPacket UavvEnableIRIsotherm::Encode()
{
	unsigned char data[2];
    if (Status == IsothermStatus::Enable)
	{
		data[0] = 1;
		data[1] = 1;
	}
	else
	{
		data[0] = 0;
		data[1] = 0;
	}
    return GimbalPacket(UavvGimbalProtocol::EnableIsotherm, data, sizeof(data));
}

ParseResult UavvEnableIRIsotherm::TryParse(GimbalPacket packet, UavvEnableIRIsotherm *EnableIRIsotherm)
{
    if (packet.Data.size() < EnableIRIsotherm->Length)
	{
        return ParseResult::InvalidLength;
	}

	IsothermStatus _status;
	if ((packet.Data[0] == 0x00) && (packet.Data[1] == 0x00))
        _status = IsothermStatus::Disable;
	else if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x01))
	{
        _status = IsothermStatus::Enable;
	}
	else
	{
        return ParseResult::InvalidData;
	}

    *EnableIRIsotherm = UavvEnableIRIsotherm(_status);
    return ParseResult::Success;
}
