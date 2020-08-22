#include<iostream>
#include"UavSetIRPalette.h"

UavvSetIRPalette::UavvSetIRPalette()
{
    setMode(PaletteMode::Whitehot);
}

UavvSetIRPalette::UavvSetIRPalette(PaletteMode mode)
{
    setMode(mode);
}

UavvSetIRPalette::~UavvSetIRPalette(){}

GimbalPacket UavvSetIRPalette::Encode()
{
	unsigned char data[2];
	data[0] = 0x00;
	switch (getMode())
	{
    case PaletteMode::Whitehot:
		data[1] = 0x00;
		break;
    case PaletteMode::Blackhot:
		data[1] = 0x01;
		break;
    case PaletteMode::Fusion:
		data[1] = 0x02;
		break;
    case PaletteMode::Rain:
		data[1] = 0x03;
		break;
	default:
		break;
	}
    return GimbalPacket(UavvGimbalProtocol::SetIRPalette, data, sizeof(data));
}

ParseResult UavvSetIRPalette::TryParse(GimbalPacket packet, UavvSetIRPalette *SetIRPalette)
{
    if (packet.Data.size() < SetIRPalette->Length)
	{
        return ParseResult::InvalidLength;
	}

	PaletteMode _mode;
	if (packet.Data[0] != 0x00)
        return ParseResult::InvalidData;
	switch (packet.Data[1])
	{
	case 0x00:
        _mode = PaletteMode::Whitehot;
		break;
	case 0x01:
        _mode = PaletteMode::Blackhot;
		break;
	case 0x02:
        _mode = PaletteMode::Fusion;
		break;
	case 0x03:
        _mode = PaletteMode::Rain;
		break;
	default:
        return ParseResult::InvalidData;
	}

    *SetIRPalette = UavvSetIRPalette(_mode);
    return ParseResult::Success;
};
