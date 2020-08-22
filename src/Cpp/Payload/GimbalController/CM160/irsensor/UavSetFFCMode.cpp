#include<iostream>
#include"UavSetFFCMode.h"

UavvSetFFCMode::UavvSetFFCMode(){}

UavvSetFFCMode::UavvSetFFCMode(FFCMode mode)
{
	setMode(mode);
};

UavvSetFFCMode::~UavvSetFFCMode(){}

GimbalPacket UavvSetFFCMode::Encode()
{
	unsigned char data[2];
    if (getMode() == FFCMode::Sensor)
	{
		data[0] = 0x01;
		data[1] = 0x01;
	}
	else
	{
		data[0] = 0x00;
		data[1] = 0x01;
	}
    return GimbalPacket(UavvGimbalProtocol::SetIRFCCMode, data, sizeof(data));
}

ParseResult UavvSetFFCMode::TryParse(GimbalPacket packet, UavvSetFFCMode *SetFFCMode)
{
    if (packet.Data.size() < SetFFCMode->Length)
	{
        return ParseResult::InvalidLength;
	}

	FFCMode _mode;
	if ((packet.Data[0] == 0x00) && (packet.Data[1] == 0x00))
        _mode = FFCMode::Operator;
	else if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x01))
        _mode = FFCMode::Sensor;
	else
        return ParseResult::InvalidData;

    *SetFFCMode = UavvSetFFCMode(_mode);
    return ParseResult::Success;
};
