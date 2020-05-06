#include "UavvOverlay.h"


UavvOverlay::UavvOverlay() {}
UavvOverlay::~UavvOverlay() {}

ParseResult UavvOverlay::TryParse(GimbalPacket packet, UavvOverlay *Overlay)
{
    if (packet.Data.size() < Overlay->Length)
    {
        return ParseResult::InvalidLength;
    }
    Overlay->Reserved = packet.Data[0];
    Overlay->EnableGlobalOverlays=(packet.Data[1]&0x80)!=0?true:false;
    Overlay->EnableLaserArm=(packet.Data[1]&0x02)!=0?true:false;
    Overlay->EnableLimitWarning =(packet.Data[1]&0x01)!=0?true:false;
    Overlay->EnableGyroStabilisationMode =(packet.Data[2]&0x80)!=0?true:false;
    Overlay->EnableGimbalMode =(packet.Data[2]&0x40)!=0?true:false;
    Overlay->EnableTrackingBoxes = (packet.Data[2]&0x20)!=0?true:false;
    Overlay->EnableHorizontalField = (packet.Data[2]&0x10)!=0?true:false;
    Overlay->EnableSlantRange = (packet.Data[2]&0x08)!=0?true:false;
    Overlay->EnableTargetLocation = (packet.Data[2]&0x04)!=0?true:false;
    Overlay->EnableTimeStamp = (packet.Data[2]&0x02)!=0?true:false;
    Overlay->EnableCrosshair = (packet.Data[2]&0x01)!=0?true:false;
    return ParseResult::Success;
}

GimbalPacket UavvOverlay::Encode()
{
    unsigned char data[3];
    data[0] = Reserved;
    data[1] = 0x00;
    data[2] = 0x00;
    unsigned char flag = 0x0000;
    data[1] =  EnableGlobalOverlays==true?(data[1]|0x80):data[1];
    data[1] =  EnableLaserArm==true?(data[1]|0x02):data[1];
    data[1] =  EnableLimitWarning ==true?(data[1]|0x01):data[1];
    data[2] =  EnableGyroStabilisationMode ==true?(data[2]|0x80):data[2];
    data[2] =  EnableGimbalMode ==true? (data[2]|0x40):data[2];
    data[2] =  EnableTrackingBoxes ==true? (data[2]|0x20):data[2];
    data[2] =  EnableHorizontalField ==true? (data[2]|0x10):data[2];
    data[2] =  EnableSlantRange ==true? (data[2]|0x08):data[2];
    data[2] =  EnableTargetLocation ==true? (data[2]|0x04):data[2];
    data[2] =  EnableTimeStamp ==true? (data[2]|0x02):data[2];
    data[2] =  EnableCrosshair ==true? (data[2]|0x01):data[2];
//    ByteManipulation::ToBytes(flag,Endianness::Little,data,1);
    return GimbalPacket(UavvGimbalProtocol::SetOverlayMode, data, sizeof(data));
}
