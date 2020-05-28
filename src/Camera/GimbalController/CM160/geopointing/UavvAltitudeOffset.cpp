#include "UavvAltitudeOffset.h"


UavvAltitudeOffset::UavvAltitudeOffset() {}
UavvAltitudeOffset::~UavvAltitudeOffset() {}

UavvAltitudeOffset::UavvAltitudeOffset(short altitude)
{
	setAltitude(altitude);
}

ParseResult UavvAltitudeOffset::TryParse(GimbalPacket packet, UavvAltitudeOffset*AltitudeOffset)
{
    if (packet.Data.size() < AltitudeOffset->Length)
	{
        return ParseResult::InvalidLength;
	}
    AltitudeOffset->Reserved = packet.Data[0];
    AltitudeOffset->Altitude = ByteManipulation::ToInt16(packet.Data.data(),1,Endianness::Big);
	return ParseResult::Success;
}

GimbalPacket UavvAltitudeOffset::Encode()
{
    unsigned char data[3];

    data[0] = Reserved;
    ByteManipulation::ToBytes((short)Altitude,Endianness::Big,data,1);
	return GimbalPacket(UavvGimbalProtocol::AltitudeOffset, data, sizeof(data));
}
