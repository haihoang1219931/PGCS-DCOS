#ifndef UAVZOOMPOSITIONRESPONSE_H
#define UAVZOOMPOSITIONRESPONSE_H

#include"../UavvPacket.h"

class UavvZoomPositionResponse
{
public:
    unsigned int Length = 2;
	unsigned short ZoomPositionResponse;
    UavvZoomPositionResponse();
    UavvZoomPositionResponse(unsigned short _ZoomPositionResponse);
    ~UavvZoomPositionResponse();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvZoomPositionResponse *ZoomPositionResponse);
};

#endif
