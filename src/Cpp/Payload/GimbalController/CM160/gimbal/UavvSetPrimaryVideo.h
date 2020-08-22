#ifndef UAVVSETPRIMARYVIDEO_H
#define UAVVSETPRIMARYVIDEO_H
#include "../UavvPacket.h"

enum class PrimaryVideoSensorType
{
    Sensor1video,
    Sensor2video,
};
class UavvSetPrimaryVideo
{
public:
    unsigned int Length = 2;
    unsigned char Data01 = 1;
    unsigned char PrimaryVideoSensor;
    UavvSetPrimaryVideo(PrimaryVideoSensorType setprimaryvideo);
	UavvSetPrimaryVideo();
	~UavvSetPrimaryVideo();
	static ParseResult TryParse(GimbalPacket packet, UavvSetPrimaryVideo *setprimary);
    GimbalPacket Encode();

};
#endif
