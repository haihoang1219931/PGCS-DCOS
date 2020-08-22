#include "UavvCurrentGimbalPositionRate.h"


UavvCurrentGimbalPositionRate::UavvCurrentGimbalPositionRate(){}
UavvCurrentGimbalPositionRate::~UavvCurrentGimbalPositionRate(){}

UavvCurrentGimbalPositionRate::UavvCurrentGimbalPositionRate(float panPosition, float tiltPosition, float panVelocity, float tiltVelocity)
{
    PanVelocity = panPosition;
    TiltPosition = tiltPosition;
    PanVelocity = panVelocity;
    TiltVelocity = tiltVelocity;
}

ParseResult UavvCurrentGimbalPositionRate::TryParse(GimbalPacket packet, UavvCurrentGimbalPositionRate *result)
{
    if (packet.Data.size() < result->Length)
	{
        return ParseResult::InvalidLength;
	}
    float panPosition, tiltPosition, panVelocity, tiltVelocity;
    panPosition  = UavvPacketHelper::PacketToAngle(packet.Data[0],packet.Data[1]);
    tiltPosition = UavvPacketHelper::PacketToAngle(packet.Data[2],packet.Data[3]);
    panVelocity  = UavvPacketHelper::PacketToVelocity(packet.Data[4],packet.Data[5]);
    tiltVelocity = UavvPacketHelper::PacketToVelocity(packet.Data[6],packet.Data[7]);
    result->PanPosition = panPosition;
    result->TiltPosition = tiltPosition;
    result->PanVelocity = panVelocity;
    result->TiltVelocity = tiltVelocity;
    //printf("panPosition: %f\r\n",panPosition);
    //printf("tiltPosition: %f\r\n",tiltPosition);
    return ParseResult::Success;
}
