#ifndef UAVVMODIFYOBJECTTRACK_H
#define UAVVMODIFYOBJECTTRACK_H

#include "../UavvPacket.h"

class UavvModifyObjectTrack
{
public:
    unsigned int Length = 5;
    ActionType action;
    unsigned short Column, Row;

    void setActModifyObjectTrack(ActionType act)
	{
        action = act;
	}

	void setRowModifyObjectTrack(unsigned int row)
	{
		Row = row;
	}

	void setColumnModifyObjectTrack(unsigned int col)
	{
		Column = col;
	}
	
    ActionType getActModifyObjectTrack()
	{
        return action;
	}

	unsigned int getRowModifyObjectTrack()
	{
		return Row;
	}

	unsigned int getColumnModifyObjectTrack()
	{
		return Column;
	}
	UavvModifyObjectTrack();
	~UavvModifyObjectTrack();
    UavvModifyObjectTrack(ActionType action, unsigned int column, unsigned int row);
	static ParseResult TryParse(GimbalPacket packet, UavvModifyObjectTrack *ModifyObjectTrack);
    GimbalPacket Encode();
};
#endif
