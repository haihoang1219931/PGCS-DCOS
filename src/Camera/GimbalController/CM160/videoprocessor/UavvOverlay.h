#ifndef UAVVOVERLAY_H
#define UAVVOVERLAY_H

#include "../UavvPacket.h"


class UavvOverlay
{
public:
    unsigned int Length = 3;
    unsigned char Reserved = 0;
    bool EnableGlobalOverlays = true;
    bool EnableLaserArm = true;
    bool EnableLimitWarning = true;
    bool EnableGyroStabilisationMode = true;
    bool EnableGimbalMode = true;
    bool EnableTrackingBoxes = true;
    bool EnableHorizontalField = true;
    bool EnableSlantRange = true;
    bool EnableTargetLocation = true;
    bool EnableTimeStamp = true;
    bool EnableCrosshair = true;

public:
    UavvOverlay();
    ~UavvOverlay();
    static ParseResult TryParse(GimbalPacket packet, UavvOverlay *Overlay);
    GimbalPacket Encode();
};
#endif
