#ifndef UAVVNUDGETRACK_H
#define UAVVNUDGETRACK_H

#include "../UavvPacket.h"

class UavvNudgeTrack
{
public:
    unsigned int Length = 3;
	char Column, Row;

	void setColumnPixelOffset(char column)
	{
		Column= column;
	}

	void setRowPixelOffset(char row)
	{
		Row = row;
	}

	unsigned char getColumnPixelOffset()
	{
		return Column;
	}

	unsigned int getRowPixelOffset()
	{
		return Row;
	}
public:
	UavvNudgeTrack();
	~UavvNudgeTrack();
	UavvNudgeTrack(char column, char row);
	static ParseResult TryParse(GimbalPacket packet, UavvNudgeTrack *NudgeTrack);
    GimbalPacket Encode();
};
#endif
