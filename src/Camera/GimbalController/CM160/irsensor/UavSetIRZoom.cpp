#include<iostream>
#include"UavSetIRZoom.h"

UavvSetIRZoom::UavvSetIRZoom(){}

UavvSetIRZoom::UavvSetIRZoom(ZoomFlag flag)
{
    Flag = flag;
}

UavvSetIRZoom::~UavvSetIRZoom(){}

GimbalPacket UavvSetIRZoom::Encode()
{
	unsigned char data[2];
    switch (Flag)
	{
    case ZoomFlag::x1:
		data[0] = 0x00;
		break;
    case ZoomFlag::x1Freeze:
		data[0] = 0x01;
		break;
    case ZoomFlag::x2:
		data[0] = 0x04;
		break;
    case ZoomFlag::x2Freeze:
		data[0] = 0x05;
		break;
    case ZoomFlag::x4:
		data[0] = 0x08;
		break;
    case ZoomFlag::x4Freeze:
		data[0] = 0x09;
		break;
    case ZoomFlag::x8:
		data[0] = 0x10;
		break;
    case ZoomFlag::x8Freeze:
		data[0] = 0x11;
		break;
	default:
		break;
	}
    data[1] = 0;
    return GimbalPacket(UavvGimbalProtocol::SetIRZoom, data, sizeof(data));
}

ParseResult UavvSetIRZoom::TryParse(GimbalPacket packet, UavvSetIRZoom *SetIRZoom)
{
    if (packet.Data.size() < SetIRZoom->Length)
	{
        return ParseResult::InvalidLength;
	}

	ZoomFlag flag;
	switch (packet.Data[0])
	{
	case 0x00:
        flag = ZoomFlag::x1;
		break;
	case 0x01:
        flag = ZoomFlag::x1Freeze;
		break;
	case 0x04:
        flag = ZoomFlag::x2;
		break;
	case 0x05:
        flag = ZoomFlag::x2Freeze;
		break;
	case 0x08:
        flag = ZoomFlag::x4;
		break;
	case 0x09:
        flag = ZoomFlag::x4Freeze;
		break;
	case 0x10:
        flag = ZoomFlag::x8;
		break;
	case 0x11:
        flag = ZoomFlag::x8Freeze;
		break;
	default:
        return ParseResult::InvalidData;
	}

	*SetIRZoom = UavvSetIRZoom(flag);
    return ParseResult::Success;
};
