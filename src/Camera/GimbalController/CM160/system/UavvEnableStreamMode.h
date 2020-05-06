#ifndef UAVVENABLESTREAMMODE_H
#define UAVVENABLESTREAMMODE_H

#include "../UavvPacket.h"

enum class EnableStreamTypeActionFlag
{
    stopStream,
    Mode1,
    Mode2
};
enum class EnableStreamFrequencyFlag
{
    stream20HzBackWards = 1,
    Stream5Hz=5,
    Stream10Hz=10,
    Stream20Hz=20,
    Stream60Hz=60,
    Stream80Hz=80,
    Stream100Hz=100,
};
class UavvEnableStreamMode
{
public:
    unsigned int Length = 2;
	unsigned char EnableStreamTypeAction;
	unsigned char EnableStreamFrequency;
	UavvEnableStreamMode(EnableStreamTypeActionFlag type, EnableStreamFrequencyFlag frequency) ;
	~UavvEnableStreamMode() ;
	UavvEnableStreamMode();
	static ParseResult TryParse(GimbalPacket packet, UavvEnableStreamMode *EnableStreamMode);
	GimbalPacket encode();
};
#endif
