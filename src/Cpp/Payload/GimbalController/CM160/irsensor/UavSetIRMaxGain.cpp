#include"UavSetIRMaxGain.h"

UavvSetIRMaxGain::UavvSetIRMaxGain(){}

UavvSetIRMaxGain::UavvSetIRMaxGain(unsigned short gain)
{
	setMaxGain(gain);
};

UavvSetIRMaxGain::~UavvSetIRMaxGain(){}

GimbalPacket UavvSetIRMaxGain::Encode()
{
	unsigned char data[2];
    ByteManipulation::ToBytes(getMaxGain(),Endianness::Big,data,0);
    return GimbalPacket(UavvGimbalProtocol::SetIRMAXAGC, data, sizeof(data));
}

ParseResult UavvSetIRMaxGain::TryParse(GimbalPacket packet, UavvSetIRMaxGain *SetIRMaxGain)
{
    if (packet.Data.size() < SetIRMaxGain->Length)
	{
        return ParseResult::InvalidLength;
	}

    unsigned short _gain = ByteManipulation::ToUInt16(packet.Data.data(),0,Endianness::Big);
	if (_gain > 2048)
        return ParseResult::InvalidData;

    *SetIRMaxGain = UavvSetIRMaxGain(_gain);
    return ParseResult::Success;
};
