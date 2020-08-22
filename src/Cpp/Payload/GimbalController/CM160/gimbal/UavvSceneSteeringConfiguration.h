#ifndef UAVVSCENESTEERINGCONFIGURATION_H
#define UAVVSCENESTEERINGCONFIGURATION_H
#include "../UavvPacket.h"

enum class SceneSteeringAction
{
    EnableOject = 0x01,
    EnableGeolockExit = 0x02,
    EnableZeroRateDemand = 0x04,
    EnablePositionMoveCompletion = 0x08
};
class UavvSceneSteeringConfiguration
{
public:
    unsigned int Length = 1;
	unsigned char sceneSteeringAction;
	UavvSceneSteeringConfiguration(SceneSteeringAction sceneaction);
	UavvSceneSteeringConfiguration();
	~UavvSceneSteeringConfiguration();
	static ParseResult TryParse(GimbalPacket packet, UavvSceneSteeringConfiguration *setprimary);
    GimbalPacket Encode();

};
#endif
