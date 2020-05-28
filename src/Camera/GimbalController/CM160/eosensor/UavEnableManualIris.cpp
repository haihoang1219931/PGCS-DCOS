#include"UavEnableManualIris.h"

UavvEnableManualIris::UavvEnableManualIris()
{
    Status = ManualIrisStatus::Enable;
}
UavvEnableManualIris::UavvEnableManualIris(bool enable){
    if (enable) {
        Status = ManualIrisStatus::Enable;
    }else{
        Status = ManualIrisStatus::Disable;
    }
}

UavvEnableManualIris::UavvEnableManualIris(ManualIrisStatus status)
{
    Status = status;
}

UavvEnableManualIris::~UavvEnableManualIris(){}

GimbalPacket UavvEnableManualIris::Encode()
{
	unsigned char data[2];
    if (Status == ManualIrisStatus::Enable)
	{
		data[0] = 1;
		data[1] = 1;
	}
    return GimbalPacket(UavvGimbalProtocol::EnableManualIris, data, sizeof(data));
}

ParseResult UavvEnableManualIris::TryParse(GimbalPacket packet, UavvEnableManualIris *EnableManualIris)
{
    if (packet.Data.size() < EnableManualIris->Length)
	{
        return ParseResult::InvalidLength;
	}
    ManualIrisStatus enableManualIris;
	if ((packet.Data[0] == 0x01)&&(packet.Data[1]==0x01))
	{
        enableManualIris = ManualIrisStatus::Enable;
	}
	else
        return ParseResult::InvalidData;
    *EnableManualIris = UavvEnableManualIris(enableManualIris);
    return ParseResult::Success;
};
