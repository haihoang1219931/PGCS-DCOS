#include "LaserRange.h"

LaserRange::LaserRange(){

}

LaserRange::~LaserRange(){

}

ParseResult LaserRange::TryParse(GimbalPacket packet, LaserRange *result){
    if(packet.Data.size() < result->Length){
        return ParseResult::InvalidLength;
    }

    result->StatusFlag = packet.Data[0];
    result->Confidence = packet.Data[1];
    result->Range = ByteManipulation::ToUInt32(packet.Data.data(),2,Endianness::Big);
//    printf("result->StatusFlag = %02x\r\n",packet.Data[0]);
//    printf("result->Confidence = %02x\r\n",packet.Data[1]);
//    printf("result->Range = %d\r\n",result->Range);
    return ParseResult::Success;
}

GimbalPacket LaserRange::Encode(){
    unsigned char data[6];
    data[0] = StatusFlag;
    data[1] = Confidence;
    ByteManipulation::ToBytes(Range,Endianness::Big,data,2);
    return GimbalPacket(UavvGimbalProtocol::LaserRange, data, sizeof(data));
}
