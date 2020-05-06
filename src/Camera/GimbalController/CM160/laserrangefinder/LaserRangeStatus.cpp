#include "LaserRangeStatus.h"

LaserRangeStatus::LaserRangeStatus()
{

}
LaserRangeStatus::~LaserRangeStatus(){

}

ParseResult LaserRangeStatus::TryParse(GimbalPacket packet, LaserRangeStatus *result){
    if(packet.Data.size() < result->Length){
        return ParseResult::InvalidLength;
    }
    result->Initialized = packet.Data[0];
    result->RangeMode = packet.Data[1];
    return ParseResult::Success;
}

GimbalPacket LaserRangeStatus::Encode(){
    unsigned char data[2];
    data[0] = Initialized;
    data[1] = RangeMode;
    return GimbalPacket(UavvGimbalProtocol::LaserRangeStatus, data, sizeof(data));
}
