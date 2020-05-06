#ifndef UAVVSTOWSTATUSRESPONSE_H
#define UAVVSTOWSTATUSRESPONSE_H
#include "../UavvPacket.h"

enum class StowModeResponse
{
    NotInStow = 0,
    InStow = 1
};
class UavvStowStatusResponse
{
public:
    unsigned int Length = 2;
    unsigned char StowMode;
    unsigned char Reserverd;
    UavvStowStatusResponse(StowModeResponse stowmode);
    ~UavvStowStatusResponse();
    UavvStowStatusResponse();
    static ParseResult TryParse(GimbalPacket packet, UavvStowStatusResponse *StatusResponse);
	GimbalPacket encode();
};

#endif
