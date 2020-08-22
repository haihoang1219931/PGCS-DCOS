#include"UavSetIRAGCMode.h"

UavvSetIRAGCMode::UavvSetIRAGCMode(){}

UavvSetIRAGCMode::UavvSetIRAGCMode(AGCMode mode)
{
    Mode = mode;
}

UavvSetIRAGCMode::~UavvSetIRAGCMode(){}

GimbalPacket UavvSetIRAGCMode::Encode()
{
	unsigned char data[2];
	data[0] = 0x01;
    switch (Mode)
	{
    case AGCMode::Auto:
		data[1] = 0x00;
		break;
    case AGCMode::OnceBright:
		data[1] = 0x01;
		break;
    case AGCMode::AutoBright:
		data[1] = 0x02;
		break;
    case AGCMode::Manual:
		data[1] = 0x03;
		break;
    case AGCMode::Linear:
		data[1] = 0x05;
		break;
	default:
		break;
	}
    return GimbalPacket(UavvGimbalProtocol::SetIRAGCMode, data, sizeof(data));
}

ParseResult UavvSetIRAGCMode::TryParse(GimbalPacket packet, UavvSetIRAGCMode *SetIRAGCMode)
{
    if (packet.Data.size() < SetIRAGCMode->Length)
	{
        return ParseResult::InvalidLength;
	}

	AGCMode mode;
	if (packet.Data[0] != 0x01)
        return ParseResult::InvalidData;
	switch (packet.Data[1])
	{
	case 0x00:
        mode = AGCMode::Auto;
		break;
	case 0x01:
        mode = AGCMode::OnceBright;
		break;
	case 0x02:
        mode = AGCMode::AutoBright;
		break;
	case 0x03:
        mode = AGCMode::Manual;
		break;
	case 0x05:
        mode = AGCMode::Linear;
		break;
	default:
        return ParseResult::InvalidData;
	}

    *SetIRAGCMode = UavvSetIRAGCMode(mode);
    return ParseResult::Success;
}
