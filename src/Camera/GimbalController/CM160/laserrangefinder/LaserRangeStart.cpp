#include "LaserRangeStart.h"

LaserRangeStart::LaserRangeStart()
{

}
LaserRangeStart::~LaserRangeStart(){

}
ParseResult LaserRangeStart::TryParse(GimbalPacket packet, LaserRangeStart *result){
    if(packet.Data.size() < result->Length){
        return ParseResult::InvalidLength;
    }
    result->Mesuaring = packet.Data[0];
    result->Reserved = ByteManipulation::ToUInt32(packet.Data.data(),1,Endianness::Big);
    return ParseResult::Success;
}
GimbalPacket LaserRangeStart::Encode(){
    unsigned char data[5];
    data[0] = Mesuaring;
    ByteManipulation::ToBytes(Reserved,Endianness::Big,data,1);
    return GimbalPacket(UavvGimbalProtocol::LaserRangeStart, data, sizeof(data));
}
