#include<iostream>
#include"UavDefog.h"

UavvDefog::UavvDefog(){}

UavvDefog::UavvDefog(FlagFog flag, StrengthFog strength)
{
    Flag = flag;
    Strength = strength;
}

UavvDefog::~UavvDefog(){}

GimbalPacket UavvDefog::Encode()
{
	unsigned char data[2];
    FlagFog _flag = Flag;
    if (_flag == FlagFog::FDisable)
	{
		data[0] = 0;
		data[1] = 0;
	}
    else if (_flag==FlagFog::EnableAuto)
	{
		data[0] = 1;
		data[1] = 0;
	}
	else
	{
		data[0] = 2;
        data[1] = (unsigned char)Strength;
	}
    return GimbalPacket(UavvGimbalProtocol::SensorDefog, data, sizeof(data));
}

ParseResult UavvDefog::TryParse(GimbalPacket packet, UavvDefog *Defog)
{
    if (packet.Data.size() < Defog->Length)
	{
        return ParseResult::InvalidLength;
	}
    FlagFog flag;
    StrengthFog strength;
	if ((packet.Data[0] == 0x00))
	{
        flag = FlagFog::FDisable;
	}
	else if ((packet.Data[0] == 0x01))
	{
        flag = FlagFog::EnableAuto;
	}
	else
	{
        flag = FlagFog::EnableManual;
		switch (packet.Data[1])
		{
		case 0x00:
            strength = StrengthFog::SDisable;
			break;
		case 0x01:
            strength = StrengthFog::Low;
			break;
		case 0x02:
            strength = StrengthFog::Medium;
			break;
		case 0x03:
            strength = StrengthFog::High;
			break;
		default:
            return ParseResult::InvalidData;
		}
	}
    *Defog = UavvDefog(flag, strength);
    return ParseResult::Success;
}
