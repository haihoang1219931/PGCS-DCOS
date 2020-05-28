#include<iostream>
#include"UavSetIRIsothermThresholds.h"

UavvSetIRIsothermThresholds::UavvSetIRIsothermThresholds(){}

UavvSetIRIsothermThresholds::UavvSetIRIsothermThresholds(unsigned char upperThreshold, unsigned char lowerThreshold)
{
	setUpperThreshold(upperThreshold);
	setLowerThreshold(lowerThreshold);
}

UavvSetIRIsothermThresholds::~UavvSetIRIsothermThresholds(){}

GimbalPacket UavvSetIRIsothermThresholds::Encode()
{
	unsigned char data[2];
	data[0] = getUpperThreshold();
	data[1] = getLowerThreshold();
    return GimbalPacket(UavvGimbalProtocol::SetIsothermThresholds, data, sizeof(data));
}

ParseResult UavvSetIRIsothermThresholds::TryParse(GimbalPacket packet, UavvSetIRIsothermThresholds *SetIRIsothermThresholds)
{
    if (packet.Data.size() < SetIRIsothermThresholds->Length)
	{
        return ParseResult::InvalidLength;
	}

	unsigned char _upperThreshold, _lowerThreshold;
	_upperThreshold = packet.Data[0];
	_lowerThreshold = packet.Data[1];

    *SetIRIsothermThresholds = UavvSetIRIsothermThresholds(_upperThreshold, _lowerThreshold);
    return ParseResult::Success;
}
