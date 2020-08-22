#ifndef UAVVCURRENTGEOLOCKSETPOINT_H
#define UAVVCURRENTGEOLOCKSETPOINT_H

#include "../UavvPacket.h"

class UavvCurrentGeolockSetpoint
{
public:
    unsigned int Length = 10;
    float Latitude;
    float Longtitude;
    float Altitude;
public:
	UavvCurrentGeolockSetpoint();
	~UavvCurrentGeolockSetpoint();
    UavvCurrentGeolockSetpoint(float latitude, float longitude, float height);
	static ParseResult TryParse(GimbalPacket packet, UavvCurrentGeolockSetpoint *CurrentGeolockSetpoint);
    GimbalPacket Encode();
};
#endif
