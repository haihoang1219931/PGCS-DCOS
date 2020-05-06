#include<iostream>
#include"UavDisableInfraredCutFilter.h"

UavvDisableInfraredCutFilter::UavvDisableInfraredCutFilter(){}
UavvDisableInfraredCutFilter::UavvDisableInfraredCutFilter(bool disable){
    if (disable) {
        Status = InfraredCutStatus::Disable;
    }else{
        Status = InfraredCutStatus::Enable;
    }
}

UavvDisableInfraredCutFilter::UavvDisableInfraredCutFilter(InfraredCutStatus status)
{
    Status = status;
}

UavvDisableInfraredCutFilter::~UavvDisableInfraredCutFilter(){}

GimbalPacket UavvDisableInfraredCutFilter::Encode()
{
	unsigned char data[2];
    if (Status == InfraredCutStatus::Enable)
	{
		data[0] = 1;
		data[1] = 1;
	}
	else
	{
		data[0] = 0;
		data[1] = 0;
	}
    return GimbalPacket(UavvGimbalProtocol::ICRMode, data, sizeof(data));
}

ParseResult UavvDisableInfraredCutFilter::TryParse(GimbalPacket packet, UavvDisableInfraredCutFilter *DisableInfraredCutFilter)
{
    if (packet.Data.size() < DisableInfraredCutFilter->Length)
	{
        return ParseResult::InvalidLength;
	}
    InfraredCutStatus disableInfraredCutStatus;
	if ((packet.Data[0] == 0x00) && (packet.Data[1] == 0x00))
	{
        disableInfraredCutStatus = InfraredCutStatus::Disable;
	}
	else if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x01))
	{
        disableInfraredCutStatus = InfraredCutStatus::Enable;
	}
	else
        return ParseResult::InvalidData;
    *DisableInfraredCutFilter = UavvDisableInfraredCutFilter(disableInfraredCutStatus);
    return ParseResult::Success;
};
