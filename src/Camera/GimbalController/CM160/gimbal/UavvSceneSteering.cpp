#include "UavvSceneSteering.h"

UavvSceneSteering::UavvSceneSteering() {}
UavvSceneSteering::~UavvSceneSteering() {}

UavvSceneSteering::UavvSceneSteering(unsigned char sceneSteering, unsigned char automaticScene)
{
	SceneSteering = sceneSteering;
	AutomaticScene = automaticScene;
}

ParseResult UavvSceneSteering::TryParse(GimbalPacket packet, UavvSceneSteering *scenesteering)
{
    if (packet.Data.size() < scenesteering->Length)
	{
        return ParseResult::InvalidLength;
	}
	scenesteering->SceneSteering = packet.Data[0];
	scenesteering->AutomaticScene = packet.Data[1];
    return ParseResult::Success;
}

GimbalPacket UavvSceneSteering::Encode()
{
	unsigned char data[2];
	data[0] = SceneSteering;
	data[1] = AutomaticScene;
	GimbalPacket result = GimbalPacket(UavvGimbalProtocol::SceneSteering, data, sizeof(data));
	return result;
}
