#include<iostream>
#include"UavSetDigitalZoomVelocity.h"

UavvSetDigitalZoomVelocity::UavvSetDigitalZoomVelocity(){}

UavvSetDigitalZoomVelocity::UavvSetDigitalZoomVelocity(ZoomStatus status, unsigned char velocity)
{
    Status = status;
    Velocity = velocity;
}

UavvSetDigitalZoomVelocity::~UavvSetDigitalZoomVelocity(){}

GimbalPacket UavvSetDigitalZoomVelocity::Encode()
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
    return GimbalPacket(UavvGimbalProtocol::SetCameraDigitalZoomVelocity, data, sizeof(data));
}

ParseResult UavvSetDigitalZoomVelocity::TryParse(GimbalPacket packet, UavvSetDigitalZoomVelocity *SetDigitalZoomVelocity)
{
    if (packet.Data.size() < SetDigitalZoomVelocity->Length)
	{
        return ParseResult::InvalidLength;
	}
	if ((packet.Data[0]>0x07) || (packet.Data[1] > 0x07))
        return ParseResult::InvalidData;
	unsigned char digitalZoomVelocity;
	ZoomStatus zoomStatus;
	if ((packet.Data[0] == 0x00) && (packet.Data[1] == 0x00)){
        zoomStatus = ZoomStatus::Stop;
		digitalZoomVelocity = 0;
	}
	else if (packet.Data[1]==0x00)
	{
        zoomStatus = ZoomStatus::Zoomin;
		digitalZoomVelocity = packet.Data[0];
	}
	else
	{
        zoomStatus = ZoomStatus::Zoomout;
		digitalZoomVelocity = packet.Data[1];
	}
    *SetDigitalZoomVelocity = UavvSetDigitalZoomVelocity(zoomStatus, digitalZoomVelocity);
    return ParseResult::Success;
}
