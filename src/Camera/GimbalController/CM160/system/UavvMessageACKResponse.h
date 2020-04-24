#ifndef UAVVMESSAGEACKRESPONSE_H
#define UAVVMESSAGEACKRESPONSE_H

#include "../UavvPacket.h"

class UavvMessageACKResponse{
public:
    unsigned int Length = 1;
    unsigned char PacketID;
    UavvMessageACKResponse();
    ~UavvMessageACKResponse();
    UavvMessageACKResponse(unsigned char packetID);
    static ParseResult TryParse(GimbalPacket packet, UavvMessageACKResponse *msgACKRes);
    GimbalPacket Encode();
};
#endif
