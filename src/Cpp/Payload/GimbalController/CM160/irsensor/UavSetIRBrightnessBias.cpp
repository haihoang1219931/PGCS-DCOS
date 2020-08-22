#include"UavSetIRBrightnessBias.h"

UavvSetIRBrightnessBias::UavvSetIRBrightnessBias(){}

UavvSetIRBrightnessBias::UavvSetIRBrightnessBias(short bias)
{
	setBias(bias);
}

UavvSetIRBrightnessBias::~UavvSetIRBrightnessBias(){}

GimbalPacket UavvSetIRBrightnessBias::Encode()
{
	unsigned char data[2];
    ByteManipulation::ToBytes((unsigned short)(255 + getBias()),Endianness::Big, data,0);
    return GimbalPacket(UavvGimbalProtocol::SetIRBrightnessBias, data, sizeof(data));
}

ParseResult UavvSetIRBrightnessBias::TryParse(GimbalPacket packet, UavvSetIRBrightnessBias *SetIRBrightnessBias)
{
    if (packet.Data.size() < SetIRBrightnessBias->Length)
	{
        return ParseResult::InvalidLength;
	}

    short _bias = (short)ByteManipulation::ToUInt16(packet.Data.data(),0,Endianness::Big) - 255;
	if (_bias <= 255 && _bias >= -255)
	{
        *SetIRBrightnessBias = UavvSetIRBrightnessBias(_bias);
        return ParseResult::Success;
	}
    return ParseResult::InvalidData;
};
