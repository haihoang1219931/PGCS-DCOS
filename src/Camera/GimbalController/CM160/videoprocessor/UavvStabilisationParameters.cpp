#include "UavvStabilisationParameters.h"


UavvStabilisationParameters::UavvStabilisationParameters() {}
UavvStabilisationParameters::~UavvStabilisationParameters() {}

UavvStabilisationParameters::UavvStabilisationParameters(unsigned char enable, unsigned char rate, unsigned char maxTranslation, unsigned char maxRotational, unsigned char background)
{
	setEnableStabilisationParameters(enable);
	setRateStabilisationParameters(rate);
	setMaxTranslation(maxTranslation);
	setMaxRotational(maxRotational);
	setBackground(background);
}

ParseResult UavvStabilisationParameters::TryParse(GimbalPacket packet, UavvStabilisationParameters *StabilisationParameters)
{
    if (packet.Data.size() < StabilisationParameters->Length)
	{
        return ParseResult::InvalidLength;
	}
	unsigned char enable, rate, maxTranslation, maxRotational, background;
	
	enable = packet.Data[0];
	rate = packet.Data[1];
	maxTranslation = packet.Data[2];
	maxRotational = packet.Data[3];
	background = packet.Data[4];
    *StabilisationParameters = UavvStabilisationParameters(enable,rate,maxTranslation,maxRotational,background);
	return ParseResult::Success;
}

GimbalPacket UavvStabilisationParameters::Encode()
{
	unsigned char data[5];
	data[0] = getEnableStabilisationParameters();
	data[1] = getRateStabilisationParameters();
	data[2] = getMaxTranslation();
	data[3] = getMaxRotational();
    data[4] = getBackground();
//    data[4] = 0x00;
    return GimbalPacket(UavvGimbalProtocol::SetStabilisationParameters, data, sizeof(data));
}
