#include<iostream>
#include"UavSetEOOpticalZoomVelocity.h"

UavvSetEOOpticalZoomVelocity::UavvSetEOOpticalZoomVelocity(){}

UavvSetEOOpticalZoomVelocity::UavvSetEOOpticalZoomVelocity(ZoomStatus status, unsigned char velocity)
{
    Status = status ;
    Velocity = velocity;
}

UavvSetEOOpticalZoomVelocity::~UavvSetEOOpticalZoomVelocity(){}

GimbalPacket UavvSetEOOpticalZoomVelocity::Encode()
{
	unsigned char data[2];
	switch (Status)
	{
    case ZoomStatus::Stop:
		data[0] = 0;
		data[1] = 0;
		break;
    case ZoomStatus::Zoomin:
        data[0] = Velocity;
		data[1] = 0;
		break;
    case ZoomStatus::Zoomout:
		data[0] = 0;
        data[1] = Velocity;
        break;
	}
    return GimbalPacket(UavvGimbalProtocol::SetCameraZoomVelocity, data, sizeof(data));
}

ParseResult UavvSetEOOpticalZoomVelocity::TryParse(GimbalPacket packet, UavvSetEOOpticalZoomVelocity *SetEOOpticalZoomVelocity)
{
    if (packet.Data.size() < SetEOOpticalZoomVelocity->Length)
	{
        return ParseResult::InvalidLength;
	}
	if ((packet.Data[0]>0x07) || (packet.Data[1] > 0x07))
        return ParseResult::InvalidData;
	unsigned char eoOpticalZoomVelocity;
	ZoomStatus zoomStatus;
	if ((packet.Data[0] == 0x00) && (packet.Data[1] == 0x00)){
        zoomStatus = ZoomStatus::Stop;
		eoOpticalZoomVelocity = 0;
	}
	else if (packet.Data[1] == 0x00)
	{
        zoomStatus = ZoomStatus::Zoomin;
		eoOpticalZoomVelocity = packet.Data[0];
	}
	else
	{
        zoomStatus = ZoomStatus::Zoomout;
		eoOpticalZoomVelocity = packet.Data[1];
	}
    *SetEOOpticalZoomVelocity = UavvSetEOOpticalZoomVelocity(zoomStatus, eoOpticalZoomVelocity);
    return ParseResult::Success;
}
