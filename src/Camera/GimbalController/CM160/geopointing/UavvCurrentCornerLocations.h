#ifndef UAVVCURRENTCORNERLOCATION_H
#define UAVVCURRENTCORNERLOCATION_H

#include "../UavvPacket.h"

class UavvCurrentCornerLocation
{
public:
    unsigned int Length = 42;
    short Reserved01 = 0;
    float TopLeftLatitude, TopLeftLongitude;
    short Reserved02 = 0;
    float TopRightLatitude, TopRightLongitude;
    short Reserved03 = 0;
    float BottomRightLatitude, BottomRightLongitude;
    short Reserved04 = 0;
    float BottomLeftLatitude, BottomLeftLongitude;
    short Reserved05 = 0;
    float CenterLatitude, CenterLongitude;
public:
	UavvCurrentCornerLocation();
	~UavvCurrentCornerLocation();
    UavvCurrentCornerLocation(float tlLatitude,
                              float tlLongitude,
                              float trLatitude,
                              float trLongitude,
                              float brLatitude,
                              float brLongitude,
                              float blLatitude,
                              float blLongitude);
	static ParseResult TryParse(GimbalPacket packet, UavvCurrentCornerLocation*CurrentCornerLocation);
    GimbalPacket Encode();
};
#endif
