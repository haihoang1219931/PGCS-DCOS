#ifndef UAVSENSORZOOM_H
#define UAVSENSORZOOM_H

#include"../UavvPacket.h"

class UavvSensorZoom
{

public:
    unsigned int Length = 6;
    unsigned char SensorIndex;
    unsigned char ZoomFlag;
	short ZoomValue;
    unsigned short Reserved = 0;

    UavvSensorZoom();
    UavvSensorZoom(unsigned char sensorIndex, short value);
    ~UavvSensorZoom();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSensorZoom *SensorZoom);
};

#endif
