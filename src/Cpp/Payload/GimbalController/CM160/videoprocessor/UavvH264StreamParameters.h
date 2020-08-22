#ifndef UAVVH264STREAMPARAMETERS_H
#define UAVVH264STREAMPARAMETERS_H

#include "../UavvPacket.h"

class UavvH264StreamParameters
{
public:
    unsigned int Length = 8;
    unsigned int BitRate;
	unsigned char FrameInterval, FrameStep, DownSample;
    unsigned char Reserved = 0;

    void setBitRateH264StreamParameters(unsigned int bitRate)
	{
		BitRate= bitRate;
	}

	void setIntervalH264StreamParameters(unsigned char interval)
	{
		FrameInterval = interval;
	}

	void setStepH264StreamParameters(unsigned char step)
	{
		FrameStep = step;
	}

	void setDownSampleH264StreamParameters(unsigned char downSample)
	{
		DownSample =downSample;
	}

    unsigned int getBitRateH264StreamParameters()
	{
		return BitRate;
	}

	unsigned char getIntervalH264StreamParameters()
	{
		return FrameInterval;
	}

	unsigned char getStepH264StreamParameters()
	{
		return FrameStep;
	}

	unsigned char getDownSampleH264StreamParameters()
	{
		return DownSample;
	}
public:
	UavvH264StreamParameters();
	~UavvH264StreamParameters();
    UavvH264StreamParameters(unsigned int bitRate,
                             unsigned char interval,
                             unsigned char step,
                             unsigned char downSample,
                             unsigned char reserved);
	static ParseResult TryParse(GimbalPacket packet, UavvH264StreamParameters *H264StreamParameters);
    GimbalPacket Encode();
};
#endif
