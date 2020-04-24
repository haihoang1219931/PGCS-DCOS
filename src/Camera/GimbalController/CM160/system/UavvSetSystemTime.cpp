#include "UavvSetSystemTime.h"

UavvSetSystemTime::~UavvSetSystemTime() {}
UavvSetSystemTime::UavvSetSystemTime() {}
UavvSetSystemTime::UavvSetSystemTime(unsigned char weekday, unsigned char date, unsigned char month, unsigned char year)
{
	Weekday = weekday;
	Date = date;
	Month = month;
	Year = year;
}

ParseResult UavvSetSystemTime::TryParse(GimbalPacket packet, UavvSetSystemTime *SystemTime)
{
    if (packet.Data.size() < SystemTime->Length)
	{
        return ParseResult::InvalidLength;
	}
	SystemTime->Weekday = packet.Data[0];
	SystemTime->Date = packet.Data[1];
	SystemTime->Month = packet.Data[2];
	SystemTime->Year = packet.Data[3];
    return ParseResult::Success;
}
GimbalPacket UavvSetSystemTime::encode()
{
	unsigned char data[4];
    ByteManipulation::ToBytes(second,Endianness::Little,data,0);
	return GimbalPacket(UavvGimbalProtocol::SetUnixTime, data, sizeof(data));
}
