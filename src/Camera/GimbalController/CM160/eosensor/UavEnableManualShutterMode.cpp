#include<iostream>
#include"UavEnableManualShutterMode.h"

UavvEnableManualShutterMode::UavvEnableManualShutterMode()
{
    Status = ManualShutterStatus::Enable;
}
UavvEnableManualShutterMode::UavvEnableManualShutterMode(bool enable){
    if (enable) {
        Status = ManualShutterStatus::Enable;
    }else{
        Status = ManualShutterStatus::Disable;
    }
}

UavvEnableManualShutterMode::UavvEnableManualShutterMode(ManualShutterStatus status)
{
    Status = status;
}

UavvEnableManualShutterMode::~UavvEnableManualShutterMode(){}

GimbalPacket UavvEnableManualShutterMode::Encode()
{
	unsigned char data[2];
    if (Status == ManualShutterStatus::Enable)
	{
		data[0] = 1;
		data[1] = 1;
	}
    return GimbalPacket(UavvGimbalProtocol::EnableManualShutter, data, sizeof(data));
}

ParseResult UavvEnableManualShutterMode::TryParse(GimbalPacket packet, UavvEnableManualShutterMode *EnableManualShutterMode)
{
    if (packet.Data.size() < EnableManualShutterMode->Length)
	{
        return ParseResult::InvalidLength;
	}
    ManualShutterStatus status;
	if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x01))
	{
        status = ManualShutterStatus::Enable;
	}
	else
        return ParseResult::InvalidData;
    *EnableManualShutterMode = UavvEnableManualShutterMode(status);
    return ParseResult::Success;
};
