#ifndef UAVSETIRISOTHERMTHRESHOLDS_H
#define UAVSETIRISOTHERMTHRESHOLDS_H

#include "../UavvPacket.h"

class UavvSetIRIsothermThresholds
{
public:
    unsigned int Length = 2;
	unsigned char UpperThreshold, LowerThreshold;

	void setUpperThreshold(unsigned char upperThreshold)
	{
		UpperThreshold = upperThreshold;
    }

	void setLowerThreshold(unsigned char lowerThreshold)
	{
		LowerThreshold = lowerThreshold;
    }

	unsigned char getUpperThreshold(){
		return UpperThreshold;
    }

	unsigned char getLowerThreshold(){
		return LowerThreshold;
    }

public:
    UavvSetIRIsothermThresholds();
    UavvSetIRIsothermThresholds(unsigned char upperThreshold, unsigned char lowerThreshold);
    ~UavvSetIRIsothermThresholds();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRIsothermThresholds *EnableIRIsotherm);
};

#endif
