#ifndef UAVVSEEDTERRAINHEIGHT_H
#define UAVVSEEDTERRAINHEIGHT_H

#include "../UavvPacket.h"

class UavvSeedTerrainHeight
{
public:
    unsigned int Length = 12;
    float Latitude, Longtitude;
	float Height;

    void setLatitudeSeedTerrainHeight(float latitude)
	{
		Latitude = latitude;
	}

    void setLongitudeSeedTerrainHeight(float longitude)
	{
        Longtitude = longitude;
	}

	void setHeightSeedTerrainHeight(float height)
	{
		Height = height;
	}

    float getLatitudeSeedTerrainHeight()
	{
		return Latitude;
	}

    float getLongitudeSeedTerrainHeight()
	{
        return Longtitude;
	}

	float getHeightSeedTerrainHeight()
	{
		return Height;
	}
public:
	UavvSeedTerrainHeight();
	~UavvSeedTerrainHeight();
    UavvSeedTerrainHeight(float latitude, float longitude, float height);
	static ParseResult TryParse(GimbalPacket packet, UavvSeedTerrainHeight *SeedTerrainHeight);
    GimbalPacket Encode();
};
#endif
