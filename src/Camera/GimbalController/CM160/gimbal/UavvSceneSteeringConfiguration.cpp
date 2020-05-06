#include "UavvSceneSteeringConfiguration.h"

UavvSceneSteeringConfiguration::UavvSceneSteeringConfiguration() {}
UavvSceneSteeringConfiguration::~UavvSceneSteeringConfiguration() {}

UavvSceneSteeringConfiguration::UavvSceneSteeringConfiguration(SceneSteeringAction sceneaction)
{
	sceneSteeringAction = (unsigned char)sceneaction;
}

ParseResult UavvSceneSteeringConfiguration::TryParse(GimbalPacket packet, UavvSceneSteeringConfiguration *setprimary)
{
    if (packet.Data.size() < setprimary->Length)
	{
        return ParseResult::InvalidLength;
	}
	setprimary->sceneSteeringAction = packet.Data[0];
    return ParseResult::Success;
}

GimbalPacket UavvSceneSteeringConfiguration::Encode()
{
	unsigned char data[1];
	data[0] = sceneSteeringAction;
	GimbalPacket result = GimbalPacket(UavvGimbalProtocol::SceneSteeringConfig, data, sizeof(data));
	return result;
}
