#include"UavCombinedZoomEnable.h"

UavvCombinedZoomEnable::UavvCombinedZoomEnable(){}

UavvCombinedZoomEnable::UavvCombinedZoomEnable(bool enable){
    if (enable) {
        Status = CombinedZoomStatus::Enable;
    }else{
        Status = CombinedZoomStatus::Disable;
    }
}
UavvCombinedZoomEnable::UavvCombinedZoomEnable(CombinedZoomStatus status)
{
    Status = status;
}

UavvCombinedZoomEnable::~UavvCombinedZoomEnable(){}

GimbalPacket UavvCombinedZoomEnable::Encode()
{
	unsigned char data[2];
    if (Status == CombinedZoomStatus::Enable)
	{
		data[0] = 0;
		data[1] = 0;
	}
	else
	{
		data[0] = 1;
		data[1] = 0;
	}
    return GimbalPacket(UavvGimbalProtocol::SetDigitalZoomSeparateMode, data, sizeof(data));
}

ParseResult UavvCombinedZoomEnable::TryParse(GimbalPacket packet, UavvCombinedZoomEnable *CombinedZoomEnable)
{
    if (packet.Data.size() < CombinedZoomEnable->Length)
	{
        return ParseResult::InvalidLength;
	}
    CombinedZoomStatus combinedZoomStatus;
	if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x00))
	{
        combinedZoomStatus = CombinedZoomStatus::Disable;
	}
	else if ((packet.Data[0] == 0x00) && (packet.Data[1] == 0x00))
	{
        combinedZoomStatus = CombinedZoomStatus::Enable;
	}
	else
        return ParseResult::InvalidData;
    *CombinedZoomEnable = UavvCombinedZoomEnable(combinedZoomStatus);
    return ParseResult::Success;
}
