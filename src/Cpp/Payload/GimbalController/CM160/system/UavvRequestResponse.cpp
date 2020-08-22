#include "UavvRequestResponse.h"

UavvRequestResponse::UavvRequestResponse() {}
UavvRequestResponse::~UavvRequestResponse() {}

UavvRequestResponse::UavvRequestResponse(unsigned char packetID)
{
    PacketID = packetID;
}
UavvRequestResponse::UavvRequestResponse(UavvGimbalProtocol packetID)
{
    PacketID = (unsigned char)packetID;
}
ParseResult UavvRequestResponse::TryParse(GimbalPacket packet, UavvRequestResponse *requestResponsePacket)
{
    if (packet.Data.size() < requestResponsePacket->Length)
	{
        return ParseResult::InvalidLength;
	}
    requestResponsePacket->PacketID = packet.Data[0];
    return ParseResult::Success;
}

GimbalPacket UavvRequestResponse::Encode()
{
    unsigned char data[1];
    data[0] = PacketID;
    return GimbalPacket(UavvGimbalProtocol::RequestPacket, data,sizeof(data));
}
