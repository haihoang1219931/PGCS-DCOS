#include "UavvSetGeolockLocation.h"


UavvSetGeolockLocation::UavvSetGeolockLocation() {}
UavvSetGeolockLocation::~UavvSetGeolockLocation() {}

UavvSetGeolockLocation::UavvSetGeolockLocation(GeoLockActionFlag flag, float latitude, float longitude, float height)
{
	setFlagSetGeolockLocation(flag);
	setLatitudeSetGeolockLocation(latitude);
	setLongitudeSetGeolockLocation(longitude);
	setHeightSetGeolockLocation(height);
}

ParseResult UavvSetGeolockLocation::TryParse(GimbalPacket packet, UavvSetGeolockLocation*SetGeolockLocation)
{
    if (packet.Data.size() < SetGeolockLocation->Length)
	{
        return ParseResult::InvalidLength;
	}
    switch (packet.Data[0]) {
    case 0x00:
        SetGeolockLocation->Flag = GeoLockActionFlag::DisableGeoLock;
        break;
    case 0x01:
        SetGeolockLocation->Flag = GeoLockActionFlag::EnableGeoLockAtCrossHair;
        break;
    case 0x02:
        SetGeolockLocation->Flag = GeoLockActionFlag::EnableGeoLockAtCoordinateGimbal;
        break;
    }
    SetGeolockLocation->Latitude = (float)ByteManipulation::ToInt32(packet.Data.data(),1,Endianness::Big);
    SetGeolockLocation->Longitude = (float)ByteManipulation::ToInt32(packet.Data.data(),5,Endianness::Big);
    SetGeolockLocation->Height = (float)ByteManipulation::ToInt16(packet.Data.data(),9,Endianness::Big);
	return ParseResult::Success;
}

GimbalPacket UavvSetGeolockLocation::Encode()
{
	unsigned char data[11];

    data[0] = (unsigned char)getFlagSetGeolockLocation();
    int LatEncode = (int)(Latitude/90.0 * (float)pow(2,31));
    int LonEncode = (int)(Longitude/180.0 * (float)pow(2,31));
    short HeightEncode = (short)(Height);
    ByteManipulation::ToBytes(LatEncode,Endianness::Little,data,1);
    ByteManipulation::ToBytes(LonEncode,Endianness::Little,data,5);
    ByteManipulation::ToBytes(HeightEncode,Endianness::Little,data,9);
    return GimbalPacket(UavvGimbalProtocol::SetGeolock, data, sizeof(data));
}
