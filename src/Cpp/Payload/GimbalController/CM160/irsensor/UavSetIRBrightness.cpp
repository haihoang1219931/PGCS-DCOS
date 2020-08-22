#include"UavSetIRBrightness.h"

UavvSetIRBrightness::UavvSetIRBrightness(){}

UavvSetIRBrightness::UavvSetIRBrightness(unsigned short brightness)
{
    Brightness = brightness;;
}

UavvSetIRBrightness::~UavvSetIRBrightness(){}

GimbalPacket UavvSetIRBrightness::Encode()
{
	unsigned char data[2];
    ByteManipulation::ToBytes(Brightness,Endianness::Big, data,0);
    return GimbalPacket(UavvGimbalProtocol::SetIRBrightness, data, sizeof(data));
}

ParseResult UavvSetIRBrightness::TryParse(GimbalPacket packet, UavvSetIRBrightness *SetIRBrightness)
{
    if (packet.Data.size() < SetIRBrightness->Length)
	{
        return ParseResult::InvalidLength;
	}

    unsigned short _brightness = ByteManipulation::ToUInt16(packet.Data.data(),0,Endianness::Big);
	if (_brightness>16383)
        return ParseResult::InvalidData;

    *SetIRBrightness = UavvSetIRBrightness(_brightness);
    return ParseResult::Success;
}
