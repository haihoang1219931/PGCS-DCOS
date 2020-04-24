#ifndef UAVVTRACKINGPARAMETERS_H
#define UAVVTRACKINGPARAMETERS_H

#include "../UavvPacket.h"

class UavvTrackingParameters
{
public:
    unsigned int Length = 2;
    unsigned char Acquisition;
    TrackingParametersAction Mode;

	void setAcqTrackingParameters(unsigned char acq)
	{
		Acquisition = acq;
	}

    void setModeTrackingParameters(TrackingParametersAction mode)
	{
		Mode = mode;
	}

    unsigned char getAcqTrackingParameters()
	{
		return Acquisition;
	}

    TrackingParametersAction getModeTrackingParameters()
	{
		return Mode;
	}

public:
	UavvTrackingParameters();
	~UavvTrackingParameters();
    UavvTrackingParameters(unsigned char acq, TrackingParametersAction mode);
	static ParseResult TryParse(GimbalPacket packet, UavvTrackingParameters *TrackingParameters);
    GimbalPacket Encode();
};
#endif
