#include<iostream>
#include"UavEnableAutoFocus.h"

UavvEnableAutoFocus::UavvEnableAutoFocus(){}
UavvEnableAutoFocus::UavvEnableAutoFocus(bool enable){
    if (enable) {
        Status = AutoFocusStatus::Enable;
    }else{
        Status = AutoFocusStatus::Disable;
    }
}

UavvEnableAutoFocus::UavvEnableAutoFocus(AutoFocusStatus status)
{
    Status = status;
}

UavvEnableAutoFocus::~UavvEnableAutoFocus(){}

GimbalPacket UavvEnableAutoFocus::Encode()
{
	unsigned char data[2];
    if (Status == AutoFocusStatus::Enable)
	{
		data[0] = 1;
		data[1] = 1;
	}
	else
	{
		data[0] = 0;
		data[1] = 0;
	}
    return GimbalPacket(UavvGimbalProtocol::EnableAutoFocus, data, sizeof(data));
}

ParseResult UavvEnableAutoFocus::TryParse(GimbalPacket packet, UavvEnableAutoFocus *EnableAutoFocus)
{
    if (packet.Data.size() < EnableAutoFocus->Length)
	{
        return ParseResult::InvalidLength;
	}
    AutoFocusStatus enableAutoFocusStatus;
	if ((packet.Data[0] == 0x00) && (packet.Data[1] == 0x00))
	{
        enableAutoFocusStatus = AutoFocusStatus::Disable;
	}
	else if ((packet.Data[0] == 0x01) && (packet.Data[1] == 0x01))
	{
        enableAutoFocusStatus = AutoFocusStatus::Enable;
	}
	else
        return ParseResult::InvalidData;
    *EnableAutoFocus = UavvEnableAutoFocus(enableAutoFocusStatus);
    return ParseResult::Success;
};
