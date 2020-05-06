#ifndef UAVVSTABILISEONTRACK_H
#define UAVVSTABILISEONTRACK_H

#include "../UavvPacket.h"

class UavvStabiliseOnTrack
{
public:
    unsigned int Length = 1;
    bool Enable;

    void setEnableStabiliseOnTrack(bool enable)
	{
		Enable = enable;
	}

    bool getEnableStabiliseOnTrack()
	{
		return Enable;
    }
	UavvStabiliseOnTrack();
	~UavvStabiliseOnTrack();
    UavvStabiliseOnTrack(bool enable);
	static ParseResult TryParse(GimbalPacket packet, UavvStabiliseOnTrack *StabiliseOnTrack);
    GimbalPacket Encode();
};

#endif
