#include "UavvGNSSStatus.h"


UavvGNSSStatus::UavvGNSSStatus() {}
UavvGNSSStatus::~UavvGNSSStatus() {}

UavvGNSSStatus::UavvGNSSStatus(unsigned short flagSystem, unsigned short flagFilter)
{
	setFilterGNSSStatus(flagSystem);
	setSystemGNSSStatus(flagFilter);
}

ParseResult UavvGNSSStatus::TryParse(GimbalPacket packet, UavvGNSSStatus*GNSSStatus)
{
    if (packet.Data.size() < GNSSStatus->Length)
	{
        return ParseResult::InvalidLength;
	}
    GNSSStatus->SystemFlag = ByteManipulation::ToUInt16(packet.Data.data(),0,Endianness::Big);
    GNSSStatus->FilterFlag = ByteManipulation::ToUInt16(packet.Data.data(),2,Endianness::Big);
	return ParseResult::Success;
}

GimbalPacket UavvGNSSStatus::Encode()
{
	unsigned char data[4];
    ByteManipulation::ToBytes(SystemFlag,Endianness::Big,data,0);
    ByteManipulation::ToBytes(FilterFlag,Endianness::Big,data,2);
	return GimbalPacket(UavvGimbalProtocol::ImuStatus, data, sizeof(data));
}
