#ifndef UAVVSETGEOLOCKLOCATION_H
#define UAVVSETGEOLOCKLOCATION_H

#include "../UavvPacket.h"

class UavvSetGeolockLocation
{
public:
    unsigned int Length = 11;
    GeoLockActionFlag Flag;
    float Latitude;
    float Longitude;
    float Height;

    void setFlagSetGeolockLocation(GeoLockActionFlag flag)
	{
		Flag = flag;
	}

    void setLatitudeSetGeolockLocation(float latitude)
	{
		Latitude = latitude;
	}

    void setLongitudeSetGeolockLocation(float longitude)
	{
		Longitude = longitude;
	}

    void setHeightSetGeolockLocation(float height)
	{
		Height = height;
	}

    GeoLockActionFlag getFlagSetGeolockLocation()
	{
		return Flag;
	}

    float getLatitudeSetGeolockLocation()
	{
		return Latitude;
	}

    float getLongitudeSetGeolockLocation()
	{
		return Longitude;
	}

    float getHeightSetGeolockLocation()
	{
		return Height;
	}
	UavvSetGeolockLocation();
	~UavvSetGeolockLocation();
    UavvSetGeolockLocation(GeoLockActionFlag flag, float latitude, float longitude, float height);
	static ParseResult TryParse(GimbalPacket packet, UavvSetGeolockLocation *SetGeolockLocation);
    GimbalPacket Encode();
};
#endif
