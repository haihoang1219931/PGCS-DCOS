#ifndef UAVVCURRENTIMAGESIZE_H
#define UAVVCURRENTIMAGESIZE_H

#include "../UavvPacket.h"

class UavvCurrentImageSize
{
public:
    unsigned  int Length = 5;
    unsigned char Reserved = 0;
    unsigned short FrameWidth, FrameHeight;
	void setFrameWidthCurrentImageSize(unsigned int frameWidth)
	{
		FrameWidth = frameWidth;
	}

	void setFrameHeightCurrentImageSize(unsigned int frameHeight)
	{
		FrameHeight = frameHeight;
	}

	unsigned char getFrameWidthCurrentImageSize()
	{
		return FrameWidth;
	}

	unsigned int getFrameHeightCurrentImageSize()
	{
		return FrameHeight;
	}

	UavvCurrentImageSize();
	~UavvCurrentImageSize();
    UavvCurrentImageSize(unsigned char reserved, unsigned int frameWidth, unsigned int frameHeight);
	static ParseResult TryParse(GimbalPacket packet, UavvCurrentImageSize *CurrentImageSize);
    GimbalPacket Encode();
};
#endif
