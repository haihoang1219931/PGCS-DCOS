#include<iostream>
#include"UavSetIRGainMode.h"

UavvSetIRGainMode::UavvSetIRGainMode(){}

UavvSetIRGainMode::UavvSetIRGainMode(GainMode mode)
{
	setMode(mode);
}

UavvSetIRGainMode::~UavvSetIRGainMode(){}

GimbalPacket UavvSetIRGainMode::Encode()
{
	unsigned char data[2];
    data[0] = Data01;
	switch (getMode())
	{
    case GainMode::Auto:
		data[1] = 0x00;
		break;
    case GainMode::Low:
		data[1] = 0x01;
		break;
    case GainMode::High:
		data[1] = 0x02;
		break;
	default:
		break;
	}
    return GimbalPacket(UavvGimbalProtocol::SetIRGainMode, data, sizeof(data));
}

ParseResult UavvSetIRGainMode::TryParse(GimbalPacket packet, UavvSetIRGainMode *SetIRGainMode)
{
    if (packet.Data.size() < SetIRGainMode->Length)
	{
        return ParseResult::InvalidLength;
	}

	GainMode _mode;
	if (packet.Data[0] != 0x01)
        return ParseResult::InvalidData;
	switch (packet.Data[1])
	{
	case 0x00:
        _mode = GainMode::Auto;
		break;
	case 0x01:
        _mode = GainMode::Low;
		break;
	case 0x02:
        _mode = GainMode::High;
		break;
	default:
        return ParseResult::InvalidData;
	}

    *SetIRGainMode = UavvSetIRGainMode(_mode);
    return ParseResult::Success;
}
