#include "LaserDeviceStatus.h"

LaserDeviceStatus::LaserDeviceStatus()
{

}
LaserDeviceStatus::~LaserDeviceStatus(){

}

ParseResult LaserDeviceStatus::TryParse(GimbalPacket packet, LaserDeviceStatus *result){
    if(packet.Data.size() < result->Length){
        return ParseResult::InvalidLength;
    }
    result->Reserved = packet.Data[0];
    result->Info = packet.Data[1];
    return ParseResult::Success;
}

GimbalPacket LaserDeviceStatus::Encode(){
    unsigned char data[2];
    data[0] = Reserved;
    data[1] = Info;
    return GimbalPacket(UavvGimbalProtocol::LaserDeviceStatus, data, sizeof(data));
}
