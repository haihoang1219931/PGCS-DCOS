#include "FireLaserDevice.h"

FireLaserDevice::FireLaserDevice()
{

}
FireLaserDevice::~FireLaserDevice(){

}

ParseResult FireLaserDevice::TryParse(GimbalPacket packet, FireLaserDevice *result){
    if(packet.Data.size() < result->Length){
        return ParseResult::InvalidLength;
    }
    result->Reserved = packet.Data[0];
    result->Fire = packet.Data[1];
    result->VerificationSequence = ByteManipulation::ToUInt32(packet.Data.data(),2,Endianness::Little);
    return ParseResult::Success;
}

GimbalPacket FireLaserDevice::Encode(){
    unsigned char data[6];
    data[0] = Reserved;
    data[1] = Fire;
    ByteManipulation::ToBytes(VerificationSequence,Endianness::Little,data,2);
    return GimbalPacket(UavvGimbalProtocol::FireLaserDevice, data, sizeof(data));
}
