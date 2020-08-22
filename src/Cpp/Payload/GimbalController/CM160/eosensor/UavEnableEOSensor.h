#ifndef UAVVENABLEEOSENSOR_H
#define UAVVENABLEEOSENSOR_H


#include"../UavvPacket.h"

class UavvEnableEOSensor
{
public:
    unsigned int Length = 1;
	SensorStatus Status;
    UavvEnableEOSensor();
    UavvEnableEOSensor(bool enable);
    UavvEnableEOSensor(SensorStatus status);
    ~UavvEnableEOSensor();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvEnableEOSensor *EnableEOSensor);
};

#endif
