#include "../UavvPacket.h"
enum class StowModeType
{
    ExitStow=0,
    EnterStow=1,
    ReadStowStatus=255,
};
class UavvStowMode
{
public:
    unsigned int Length = 2;
    unsigned char StowMode;
    unsigned char Reserverd;
    UavvStowMode(StowModeType stowmode);
	UavvStowMode();
	~UavvStowMode();
	static ParseResult TryParse(GimbalPacket packet, UavvStowMode *stowmode);
	GimbalPacket encode();

};
