#ifndef UAVVREQUESTRESPONSE_H
#define UAVVREQUESTRESPONSE_H

#include "../UavvPacket.h"

class UavvRequestResponse
{

public:
    unsigned int Length = 1;
    unsigned char PacketID;
    UavvRequestResponse();
    ~UavvRequestResponse();
    UavvRequestResponse(unsigned char packetID);
    UavvRequestResponse(UavvGimbalProtocol packetID);
    static ParseResult TryParse(GimbalPacket packet, UavvRequestResponse *requestResponsePacket);
	GimbalPacket Encode();
};
#endif
