#include "UavvSeedTerrainHeight.h"


UavvSeedTerrainHeight::UavvSeedTerrainHeight() {}
UavvSeedTerrainHeight::~UavvSeedTerrainHeight() {}

UavvSeedTerrainHeight::UavvSeedTerrainHeight(float latitude, float longitude, float height)
{
	setLatitudeSeedTerrainHeight(latitude);
	setLongitudeSeedTerrainHeight(longitude);
	setHeightSeedTerrainHeight(height);
}

ParseResult UavvSeedTerrainHeight::TryParse(GimbalPacket packet, UavvSeedTerrainHeight*SeedTerrainHeight)
{
    if (packet.Data.size() < SeedTerrainHeight->Length)
	{
        return ParseResult::InvalidLength;
	}
    int encodeLat = ByteManipulation::ToInt32(packet.Data.data(),1,Endianness::Little);
    int encodeLon = ByteManipulation::ToInt32(packet.Data.data(),5,Endianness::Little);
    float height = ByteManipulation::ToFloat(packet.Data.data(),9,Endianness::Little);

    SeedTerrainHeight->Latitude = encodeLat / pow(2,31) * 90.0f;
    SeedTerrainHeight->Longtitude = encodeLon  / pow(2,31) * 180.0f;
    SeedTerrainHeight->Height = height;
	return ParseResult::Success;
}

GimbalPacket UavvSeedTerrainHeight::Encode()
{
    unsigned char data[12];
    int encodeLat = (int)(Latitude / 90.0f * pow(2,31));
    int encodeLon = (int)(Longtitude / 180.0f * pow(2,31));
    ByteManipulation::ToBytes(encodeLat,Endianness::Little,data,0);
    ByteManipulation::ToBytes(encodeLon,Endianness::Little,data,4);
    ByteManipulation::ToBytes(Height,Endianness::Little,data,8);
	return GimbalPacket(UavvGimbalProtocol::SeedTerrainHeight, data, sizeof(data));
}
