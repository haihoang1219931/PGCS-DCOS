#include"UavEnableLensStabilization.h"

UavvEnableLensStabilization::UavvEnableLensStabilization(){}
UavvEnableLensStabilization::UavvEnableLensStabilization(bool enable){
    if (enable) {
        Status = LensStabilizationStatus::Enable;
    }else{
        Status = LensStabilizationStatus::Disable;
    }
}

UavvEnableLensStabilization::UavvEnableLensStabilization(LensStabilizationStatus status)
{
    Status = status;
}

UavvEnableLensStabilization::~UavvEnableLensStabilization(){}

GimbalPacket UavvEnableLensStabilization::Encode()
{
	unsigned char data[2];
    if (Status == LensStabilizationStatus::Enable)
	{
		data[0] = 1;
		data[1] = 1;
	}
	else
	{
		data[0] = 0;
		data[1] = 0;
	}
    return GimbalPacket(UavvGimbalProtocol::EnableLensStabilisation, data, sizeof(data));
}

ParseResult UavvEnableLensStabilization::TryParse(GimbalPacket packet, UavvEnableLensStabilization *EnableLensStabilization)
{
    if (packet.Data.size() < EnableLensStabilization->Length)
	{
        return ParseResult::InvalidLength;
	}
    LensStabilizationStatus lensStabilizationStatus;
	if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x01))
	{
        lensStabilizationStatus = LensStabilizationStatus::Disable;
	}
	else if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x01))
	{
        lensStabilizationStatus = LensStabilizationStatus::Enable;
	}
	else
        return ParseResult::InvalidData;
    *EnableLensStabilization = UavvEnableLensStabilization(lensStabilizationStatus);
    return ParseResult::Success;
};
