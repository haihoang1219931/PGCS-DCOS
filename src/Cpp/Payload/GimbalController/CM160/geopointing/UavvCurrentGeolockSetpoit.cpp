#include "UavvCurrentGeolockSetpoit.h"


UavvCurrentGeolockSetpoint::UavvCurrentGeolockSetpoint() {}
UavvCurrentGeolockSetpoint::~UavvCurrentGeolockSetpoint() {}

UavvCurrentGeolockSetpoint::UavvCurrentGeolockSetpoint(float latitude,
                                                       float longtitude,
                                                       float altitude){
    Latitude = latitude;
    Longtitude = longtitude;
    Altitude = altitude;
}

ParseResult UavvCurrentGeolockSetpoint::TryParse(GimbalPacket packet, UavvCurrentGeolockSetpoint*CurrentGeolockSetpoint)
{
    if (packet.Data.size() < CurrentGeolockSetpoint->Length)
	{
        return ParseResult::InvalidLength;
	}
    long lat,lon,alt;
    lat = (long)ByteManipulation::ToInt32(packet.Data.data(),0,Endianness::Big);
    lon = (long)ByteManipulation::ToInt32(packet.Data.data(),4,Endianness::Big);
    alt = (long)ByteManipulation::ToInt16(packet.Data.data(),8,Endianness::Big);

    CurrentGeolockSetpoint->Latitude = (float)((double)((double)lat + 1 -pow(2,31))  * 180 / pow(2,32))+90;
    CurrentGeolockSetpoint->Longtitude = (float)((double)((double)lon + 1 -pow(2,31))  * 360 / pow(2,32))+180;
    CurrentGeolockSetpoint->Altitude = (float)alt;
    if(CurrentGeolockSetpoint->Latitude == -90 || CurrentGeolockSetpoint->Longtitude == -90){
        return ParseResult::InvalidData;
    }
    return ParseResult::Success;
}

GimbalPacket UavvCurrentGeolockSetpoint::Encode()
{
	unsigned char data[10];

    int lat,lon;
    short alt;
    lat = (long)((Latitude - 90) * pow(2,32) / 180 + pow(2,31) -1);
    lon = (long)((Longtitude - 180) * pow(2,32) / 360 + pow(2,31) -1);
    alt = (short)Altitude;
    ByteManipulation::ToBytes(lat,Endianness::Big,data,0);
    ByteManipulation::ToBytes(lon,Endianness::Big,data,4);
    ByteManipulation::ToBytes(alt,Endianness::Big,data,8);
	return GimbalPacket(UavvGimbalProtocol::SetVideoDestination, data, sizeof(data));
}
