#ifndef UAVVSCENESTEERING_H
#define UAVVSCENESTEERING_H

#include "../UavvPacket.h"

class UavvSceneSteering
{
public:
    unsigned int Length = 2;
	unsigned char SceneSteering;
	unsigned char AutomaticScene; 
	UavvSceneSteering(unsigned char sceneSteering, unsigned char automaticScene);
	UavvSceneSteering();
	~UavvSceneSteering();
	static ParseResult TryParse(GimbalPacket packet, UavvSceneSteering *scenesteering);
    GimbalPacket Encode();

};
#endif
