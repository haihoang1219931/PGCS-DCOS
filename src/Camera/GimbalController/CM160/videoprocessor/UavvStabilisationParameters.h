#ifndef UAVVSTABILISATIONPARAMETERS_H
#define UAVVSTABILISATIONPARAMETERS_H

#include "../UavvPacket.h"

class UavvStabilisationParameters
{
public:
    unsigned int Length = 5;
	unsigned char Enable, Rate, MaxTranslation, MaxRotational, Background;

	void setEnableStabilisationParameters(unsigned char enable)
	{
		Enable = enable;
	}

	void setRateStabilisationParameters(unsigned char rate)
	{
		Rate = rate;
	}

	void setMaxTranslation(unsigned char maxTranslation)
	{
		MaxTranslation= maxTranslation;
	}

	void setMaxRotational(unsigned char maxRotational)
	{
		MaxRotational = maxRotational;
	}

	void setBackground(unsigned char background)
	{
		Background = background;
	}

	unsigned char getEnableStabilisationParameters()
	{
		return Enable;
	}

	unsigned char getRateStabilisationParameters()
	{
		return Rate;
	}

	unsigned char getMaxTranslation()
	{
		return MaxTranslation;
	}

	unsigned char getMaxRotational()
	{
		return MaxRotational;
	}

	unsigned char getBackground()
	{
		return Background;
	}

	UavvStabilisationParameters();
	~UavvStabilisationParameters();
	UavvStabilisationParameters(unsigned char enable, unsigned char rate, unsigned char maxTranslation, unsigned char maxRotational, unsigned char background);
	static ParseResult TryParse(GimbalPacket packet, UavvStabilisationParameters *StabilisationParameters);
    GimbalPacket Encode();
};
#endif
