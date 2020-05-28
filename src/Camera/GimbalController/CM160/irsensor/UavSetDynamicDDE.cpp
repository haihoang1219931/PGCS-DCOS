#include<iostream>
#include"UavSetDynamicDDE.h"

UavvSetDynamicDDE::UavvSetDynamicDDE(){}

UavvSetDynamicDDE::UavvSetDynamicDDE(ManualDDEStatus status, unsigned char sharpness)
{
	setStatus(status);
	setSharpness(sharpness);
}

UavvSetDynamicDDE::~UavvSetDynamicDDE(){}

GimbalPacket UavvSetDynamicDDE::Encode()
{
	unsigned char data[2];
    if (getStatus() == ManualDDEStatus::Enable)
	{
		data[0] = 1;
	}
	else
	{
		data[0] = 0;
	}
	data[1] = getSharpness();
    return GimbalPacket(UavvGimbalProtocol::SetIRDDE, data, sizeof(data));
}

ParseResult UavvSetDynamicDDE::TryParse(GimbalPacket packet, UavvSetDynamicDDE *SetDynamicDDE)
{
    if (packet.Data.size() < SetDynamicDDE->Length)
	{
        return ParseResult::InvalidLength;
	}

	ManualDDEStatus _status;
	unsigned char _sharpness = packet.Data[1];
	if (packet.Data[0] == 0x00)
        _status = ManualDDEStatus::Disable;
	else if (packet.Data[0] == 0x01)
	{
        _status = ManualDDEStatus::Enable;
	}
	else
	{
        return ParseResult::InvalidData;
	}

	if (_sharpness > 63)
        return ParseResult::InvalidData;

    *SetDynamicDDE = UavvSetDynamicDDE(_status, _sharpness);
    return ParseResult::Success;
};
