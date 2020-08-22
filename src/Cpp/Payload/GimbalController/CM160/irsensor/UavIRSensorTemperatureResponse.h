#ifndef UAVIRSENSORTEMPERATURERESPONSE_H
#define UAVIRSENSORTEMPERATURERESPONSE_H

#include"../UavvPacket.h"

class UavvIRSensorTemperatureResponse
{
public:
    unsigned int Length = 2;
    float Temperature = 0;

    UavvIRSensorTemperatureResponse();
    UavvIRSensorTemperatureResponse(float t);
    ~UavvIRSensorTemperatureResponse();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvIRSensorTemperatureResponse *IRSensorTemperatureResponse);
};

#endif
