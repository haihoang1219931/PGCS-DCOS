#include "UavvCurrentTargetLocation.h"


UavvCurrentTargetLocation::UavvCurrentTargetLocation() {}
UavvCurrentTargetLocation::~UavvCurrentTargetLocation() {}

UavvCurrentTargetLocation::UavvCurrentTargetLocation(unsigned char flag,
                                                     int latitude,
                                                     int longitude,
                                                     unsigned short slantRange)
{
	setFlagCurrentTargetLocation(flag);
	setLatitudeCurrentTargetLocation(latitude);
	setLongitudeCurrentTargetLocation(longitude);
	setSlantRangeCurrentTargetLocation(slantRange);
}

ParseResult UavvCurrentTargetLocation::TryParse(GimbalPacket packet, UavvCurrentTargetLocation*CurrentTargetLocation)
{
    if (packet.Data.size() < CurrentTargetLocation->Length)
	{
        return ParseResult::InvalidLength;
	}
    int lat,lon;
    unsigned int slantRange;
    CurrentTargetLocation->Flag = packet.Data[0];
    lat = ByteManipulation::ToInt32(packet.Data.data(),1,Endianness::Big);
    lon = ByteManipulation::ToInt32(packet.Data.data(),5,Endianness::Big);
    slantRange = ByteManipulation::ToUInt32(packet.Data.data(),11,Endianness::Big);
    CurrentTargetLocation->Reserved01 = ByteManipulation::ToUInt16(packet.Data.data(),9,Endianness::Big);
    CurrentTargetLocation->Reserved02 = ByteManipulation::ToUInt16(packet.Data.data(),15,Endianness::Big);
    CurrentTargetLocation->Latitude = (float)((double)((double)lat + 1 -pow(2,31))  * 180 / pow(2,32))+90;
    CurrentTargetLocation->Longitude = (float)((double)((double)lon + 1 -pow(2,31))  * 360 / pow(2,32))+180;
    CurrentTargetLocation->SlantRange = (float)slantRange*(5*pow(10,6)-1)/(pow(2,32)-1);
    if(CurrentTargetLocation->Latitude == -90 || CurrentTargetLocation->Longitude == -90){
//        printf("CurrentTargetLocation invalid\r\n");
        return ParseResult::InvalidData;
    }else{
//        printf("CurrentTargetLocation SlantRange = %d\r\n",slantRange);
    }
    return ParseResult::Success;
}

GimbalPacket UavvCurrentTargetLocation::Encode()
{
    unsigned char data[17];
    data[0] = Flag;
    int lat,lon;
    unsigned int slantRange;
    lat = (int)((Latitude - 90) * pow(2,32) / 180 + pow(2,31) -1);
    lon = (int)((Longitude - 180) * pow(2,32) / 360 + pow(2,31) -1);
    slantRange = (unsigned int)(SlantRange*(pow(2,32)-1)/(5*pow(10, 6)-1));
    ByteManipulation::ToBytes(lat,Endianness::Big,data,1);
    ByteManipulation::ToBytes(lon,Endianness::Big,data,5);
    ByteManipulation::ToBytes(Reserved01,Endianness::Big,data,9);
    ByteManipulation::ToBytes(slantRange,Endianness::Big,data,11);
    ByteManipulation::ToBytes(Reserved02,Endianness::Big,data,15);
	return GimbalPacket(UavvGimbalProtocol::SetVideoDestination, data, sizeof(data));
}
