#ifndef UAVVVIDEOCONFIGURATION_H
#define UAVVVIDEOCONFIGURATION_H

#include "../UavvPacket.h"

class UavvVideoConfiguration
{
public:
    unsigned int Length = 5;
    VideoConfigurationEncoderType EncoderType;
    VideoConfigurationOutputFrameSize Sensor0;
    VideoConfigurationOutputFrameSize Sensor1;
    unsigned short Reseved = 0;

    void setEncoderVideoConfiguration(VideoConfigurationEncoderType encoder)
	{
		EncoderType = encoder;
	}

    void setSensor0VideoConfiguration(VideoConfigurationOutputFrameSize sensor0)
	{
		Sensor0 = sensor0;
	}

    void setSensor1VideoConfiguration(VideoConfigurationOutputFrameSize sensor1)
	{
		Sensor1 = sensor1;
	}

    VideoConfigurationEncoderType getEncoderVideoConfiguration()
	{
		return EncoderType;
	}

    VideoConfigurationOutputFrameSize getSensor0VideoConfiguration()
	{
		return Sensor0;
	}

    VideoConfigurationOutputFrameSize getSensor1VideoConfiguration()
	{
		return Sensor1;
	}
public:
	UavvVideoConfiguration();
	~UavvVideoConfiguration();
    UavvVideoConfiguration(VideoConfigurationEncoderType encoder,
                           VideoConfigurationOutputFrameSize sensor0,
                           VideoConfigurationOutputFrameSize sensor1);
	static ParseResult TryParse(GimbalPacket packet, UavvVideoConfiguration *VideoConfiguration);
    GimbalPacket Encode();
};
#endif
