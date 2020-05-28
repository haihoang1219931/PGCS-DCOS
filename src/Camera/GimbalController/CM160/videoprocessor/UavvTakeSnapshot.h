#ifndef UAVVTAKESNAPSHOT_H
#define UAVVTAKESNAPSHOT_H

#include "../UavvPacket.h"

class UavvTakeSnapshot
{
public:
    unsigned int Length = 3;
    unsigned char Set = 1;
    unsigned short Reserved = 0;

	void setSetTakeSnapshot(unsigned char set)
	{
		Set = set;
	}

	unsigned char getSetTakeSnapshot()
	{
		return Set;
	}
	UavvTakeSnapshot();
	~UavvTakeSnapshot();
	UavvTakeSnapshot(unsigned char set);
	static ParseResult TryParse(GimbalPacket packet, UavvTakeSnapshot*TakeSnapshot);
    GimbalPacket Encode();
};
#endif
