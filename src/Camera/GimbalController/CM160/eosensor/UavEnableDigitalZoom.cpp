#include<iostream>
#include"UavEnableDigitalZoom.h"

UavvEnableDigitalZoom::UavvEnableDigitalZoom(){}
UavvEnableDigitalZoom::UavvEnableDigitalZoom(bool enable){
    if (enable) {
        Status = DigitalZoomStatus::Enable;
    }else{
        Status = DigitalZoomStatus::Disable;
    }
}

UavvEnableDigitalZoom::UavvEnableDigitalZoom(DigitalZoomStatus status)
{
    Status = status;
}
UavvEnableDigitalZoom::~UavvEnableDigitalZoom(){}

GimbalPacket UavvEnableDigitalZoom::Encode()
{
	unsigned char data[2];
    if (Status == DigitalZoomStatus::Enable)
	{
		data[0] = 1;
		data[1] = 0;
	}
	else
	{
		data[0] = 0;
		data[1] = 0;
	}
    return GimbalPacket(UavvGimbalProtocol::EnableDigitalZoom, data, sizeof(data));
}

ParseResult UavvEnableDigitalZoom::TryParse(GimbalPacket packet, UavvEnableDigitalZoom *EnableDigitalZoom)
{
    if (packet.Data.size() < EnableDigitalZoom->Length)
	{
        return ParseResult::InvalidLength;
	}
    DigitalZoomStatus enableDigitalZoomStatus;
	if ((packet.Data[0] == 0x00) && (packet.Data[1] == 0x00))
	{
        enableDigitalZoomStatus = DigitalZoomStatus::Disable;
	}
	else if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x00))
	{
        enableDigitalZoomStatus = DigitalZoomStatus::Enable;
	}
	else
        return ParseResult::InvalidData;
	*EnableDigitalZoom = UavvEnableDigitalZoom(enableDigitalZoomStatus);
    return ParseResult::Success;
}
