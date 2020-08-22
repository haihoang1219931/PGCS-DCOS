#include "ArmLaserDevice.h"

ArmLaserDevice::ArmLaserDevice()
{

}
ArmLaserDevice::~ArmLaserDevice(){

}

ParseResult ArmLaserDevice::TryParse(GimbalPacket packet, ArmLaserDevice *result){
    if(packet.Data.size() < result->Length){
        return ParseResult::InvalidLength;
    }
    result->Reserved = packet.Data[0];
    result->Arm = packet.Data[1];
    return ParseResult::Success;
}

GimbalPacket ArmLaserDevice::Encode(){
    unsigned char data[2];
    data[0] = Reserved;
    data[1] = Arm;
    return GimbalPacket(UavvGimbalProtocol::ArmLaserDevice, data, sizeof(data));
}
