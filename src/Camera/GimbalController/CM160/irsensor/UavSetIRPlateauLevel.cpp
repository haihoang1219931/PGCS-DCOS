#include"UavSetIRPlateauLevel.h"

UavvSetIRPlateauLevel::UavvSetIRPlateauLevel(){}

UavvSetIRPlateauLevel::UavvSetIRPlateauLevel(unsigned short level)
{
	setLevel(level);
}

UavvSetIRPlateauLevel::~UavvSetIRPlateauLevel(){}

GimbalPacket UavvSetIRPlateauLevel::Encode()
{
	unsigned char data[2];
    ByteManipulation::ToBytes(getLevel(),Endianness::Big,data,0);
    return GimbalPacket(UavvGimbalProtocol::SetIRPlateauLevel,data, sizeof(data));
}

ParseResult UavvSetIRPlateauLevel::TryParse(GimbalPacket packet, UavvSetIRPlateauLevel *SetIRPlateauLevel)
{
    if (packet.Data.size() < SetIRPlateauLevel->Length)
	{
        return ParseResult::InvalidLength;
	}

    unsigned short _level = ByteManipulation::ToUInt16(packet.Data.data(),0,Endianness::Big);
	if (_level > 1000)
        return ParseResult::InvalidData;

    *SetIRPlateauLevel = UavvSetIRPlateauLevel(_level);
    return ParseResult::Success;
};
