#include"UavSetFFCTemperatureDelta.h"

UavvSetFFCTemperatureDelta::UavvSetFFCTemperatureDelta()
{
	setTemperatureDelta(1.1);
}

UavvSetFFCTemperatureDelta::UavvSetFFCTemperatureDelta(float dt)
{
	setTemperatureDelta(dt);
}

UavvSetFFCTemperatureDelta::~UavvSetFFCTemperatureDelta(){}

GimbalPacket UavvSetFFCTemperatureDelta::Encode()
{
	unsigned char data[2];
    ByteManipulation::ToBytes((unsigned short)(10 * getTemperatureDelta()),Endianness::Big, data,0);
    return GimbalPacket(UavvGimbalProtocol::SetIRFCCTemperature, data, sizeof(data));
}

ParseResult UavvSetFFCTemperatureDelta::TryParse(GimbalPacket packet, UavvSetFFCTemperatureDelta *SetFFCTemperatureDelta)
{
    if (packet.Data.size() < SetFFCTemperatureDelta->Length)
	{
        return ParseResult::InvalidLength;
	}

    float _dt = (float)ByteManipulation::ToUInt16(packet.Data.data(),0,Endianness::Big) / 10;
	if (_dt>100)
        return ParseResult::InvalidData;
	*SetFFCTemperatureDelta = UavvSetFFCTemperatureDelta(_dt);
    return ParseResult::Success;
}
