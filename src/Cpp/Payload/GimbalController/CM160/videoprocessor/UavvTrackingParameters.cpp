#include "UavvTrackingParameters.h"


UavvTrackingParameters::UavvTrackingParameters() {}
UavvTrackingParameters::~UavvTrackingParameters() {}

UavvTrackingParameters::UavvTrackingParameters(unsigned char acq, TrackingParametersAction mode)
{
	setAcqTrackingParameters(acq);
	setModeTrackingParameters(mode);
}

ParseResult UavvTrackingParameters::TryParse(GimbalPacket packet, UavvTrackingParameters *TrackingParameters)
{
    if (packet.Data.size() < TrackingParameters->Length)
	{
        return ParseResult::InvalidLength;
	}
    unsigned char acq;
    TrackingParametersAction mode;
    if (packet.Data[1] == 0x00)
        mode = TrackingParametersAction::NoChange;
    else if (packet.Data[1] == 0x01)
        mode = TrackingParametersAction::StationaryObject;
    else if (packet.Data[1] == 0x02)
        mode = TrackingParametersAction::MovingObjects;
	else
        return ParseResult::InvalidData;
    acq = packet.Data[0];
    *TrackingParameters = UavvTrackingParameters(acq, mode);
	return ParseResult::Success;
}

GimbalPacket UavvTrackingParameters::Encode()
{
	unsigned char data[2];
    data[0] = Acquisition;
    if (Mode == TrackingParametersAction::NoChange)
        data[1] = 0x00;
    else if (Mode == TrackingParametersAction::StationaryObject)
        data[1] = 0x01;
    else if (Mode == TrackingParametersAction::MovingObjects)
        data[1] = 0x02;

    return GimbalPacket(UavvGimbalProtocol::SetTrackingParameters, data, sizeof(data));
}
