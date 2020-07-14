#include<iostream>
#include"UavEnableAutoExposure.h"

UavvEnableAutoExposure::UavvEnableAutoExposure(){}
UavvEnableAutoExposure::UavvEnableAutoExposure(bool enable){
    if(enable) {
        Status = ExposureStatus::Enable;
    }else{
        Status = ExposureStatus::Disable;
    }
}

UavvEnableAutoExposure::UavvEnableAutoExposure(ExposureStatus status)
{
    Status = status;
}

UavvEnableAutoExposure::~UavvEnableAutoExposure(){}

GimbalPacket UavvEnableAutoExposure::Encode()
{
	unsigned char data[2];
    if (Status == ExposureStatus::Enable)
	{
		data[0] = 1;
		data[1] = 1;
	}
	else
	{
		data[0] = 0;
		data[1] = 0;
	}
    return GimbalPacket(UavvGimbalProtocol::EnableAutoExposure, data, sizeof(data));
}

ParseResult UavvEnableAutoExposure::TryParse(GimbalPacket packet, UavvEnableAutoExposure *EnableAutoExposure)
{
    if (packet.Data.size() < EnableAutoExposure->Length)
	{
        return ParseResult::InvalidLength;
	}
    ExposureStatus status;
	if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x01))
	{
        status = ExposureStatus::Disable;
	}
	else if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x01))
	{
        status = ExposureStatus::Enable;
	}
	else
        return ParseResult::InvalidData;
    *EnableAutoExposure = UavvEnableAutoExposure(status);
    return ParseResult::Success;
}
