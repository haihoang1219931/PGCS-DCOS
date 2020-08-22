#include<iostream>
#include"UavZoomPositionResponse.h"

UavvZoomPositionResponse::UavvZoomPositionResponse(){}
UavvZoomPositionResponse::~UavvZoomPositionResponse(){}

UavvZoomPositionResponse::UavvZoomPositionResponse(unsigned short _ZoomPositionResponse)
{
    ZoomPositionResponse = _ZoomPositionResponse;
}

GimbalPacket UavvZoomPositionResponse::Encode()
{
	unsigned char data[2];
	ByteManipulation::ToBytes(ZoomPositionResponse,Endianness::Big, data,0);
	return GimbalPacket(UavvGimbalProtocol::ZoomPositionResponse, data, sizeof(data));
}

ParseResult UavvZoomPositionResponse::TryParse(GimbalPacket packet, UavvZoomPositionResponse *ZoomPositionResponse)
{
	if (packet.Data.size() < ZoomPositionResponse->Length)
	{
        return ParseResult::InvalidLength;
	}

	unsigned short _ZoomPositionResponse = ByteManipulation::ToUInt16(packet.Data.data(),0,Endianness::Big);
    unsigned short zoom[30] = {
        0x0000,0x2372,0x3291,0x3b83,0x41B0,0x4668,0x49FB,0x4D3C,0x5000,0x5270,
        0x548D,0x56AA,0x589E,0x5A68,0x5BD3,0x5D2B,0x5E4F,0x5F48,0x6018,0x60BF,
        0x6165,0x61E2,0x625F,0x62B2,0x6306,0x6359,0x6383,0x63AC,0x63D6,0x6400};
    if(_ZoomPositionResponse >= zoom[29]) ZoomPositionResponse->ZoomPositionResponse = 30;
    else{
        for(unsigned int i=0; i< sizeof(zoom)-1; i++){
            if(_ZoomPositionResponse>=zoom[i] && _ZoomPositionResponse < zoom[i+1]){
                ZoomPositionResponse->ZoomPositionResponse = i+1;
                break;
            }
        }
    }
    //ZoomPositionResponse->ZoomPositionResponse = (int)((float)_ZoomPositionResponse * 30.0 /64000.0);
    //printf("Zoom:0x%02x%02x = %d\r\n",packet.Data[0],packet.Data[1],(int)((float)_ZoomPositionResponse * 30.0 /64000.0));
	return ParseResult::Success;
}
