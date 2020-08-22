#ifndef UAVSENSORCURRENTFOV_H
#define UAVSENSORCURRENTFOV_H

#include"../UavvPacket.h"


class FOV{
public:
    uint8_t sensorIndex;
    float horizontal;
    float vertical;
    FOV(){}
    ~FOV(){}
};
class UavvSensorCurrentFoV
{
public:
    unsigned int Length = 10;

    vector<ImageSensorType> Type;
    vector<float> Horizontal;
    vector<float> Vertical;
    int numSensor = 0;
    UavvSensorCurrentFoV();
    UavvSensorCurrentFoV(ImageSensorType type, float hfov,float vfov);
    ~UavvSensorCurrentFoV();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSensorCurrentFoV *SensorCurrentFoV);
};

#endif
