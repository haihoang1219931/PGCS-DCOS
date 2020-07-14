#include<iostream>
#include"UavSetIRVideoOrientation.h"

UavvSetIRVideoOrientation::UavvSetIRVideoOrientation(){}

UavvSetIRVideoOrientation::UavvSetIRVideoOrientation(VideoOrientationMode mode)
{
	setMode(mode);
}

UavvSetIRVideoOrientation::~UavvSetIRVideoOrientation(){}

GimbalPacket UavvSetIRVideoOrientation::Encode()
{
	unsigned char data[2];
	data[0] = 0x00;
	switch (getMode())
	{
    case VideoOrientationMode::Normal:
		data[1] = 0x00;
		break;
    case VideoOrientationMode::Invert:
		data[1] = 0x01;
		break;
    case VideoOrientationMode::Revert:
		data[1] = 0x02;
		break;
    case VideoOrientationMode::BothIR:
		data[1] = 0x03;
		break;
	default:
		break;
	}
    return GimbalPacket(UavvGimbalProtocol::SetIRVideoOrientation, data, sizeof(data));
}

ParseResult UavvSetIRVideoOrientation::TryParse(GimbalPacket packet, UavvSetIRVideoOrientation *SetIRVideoOrientation)
{
    if (packet.Data.size() < SetIRVideoOrientation->Length)
	{
        return ParseResult::InvalidLength;
	}

	VideoOrientationMode _mode;
	if (packet.Data[0] != 0x00)
        return ParseResult::InvalidData;
	switch (packet.Data[1])
	{
	case 0x00:
        _mode = VideoOrientationMode::Normal;
		break;
	case 0x01:
        _mode = VideoOrientationMode::Invert;
		break;
	case 0x02:
        _mode = VideoOrientationMode::Revert;
		break;
	case 0x03:
        _mode = VideoOrientationMode::BothIR;
		break;
	default:
        return ParseResult::InvalidData;
	}
    *SetIRVideoOrientation = UavvSetIRVideoOrientation(_mode);
    return ParseResult::Success;
}
