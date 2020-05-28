#include "UavvMessageACKResponse.h"

UavvMessageACKResponse::UavvMessageACKResponse(){

}

UavvMessageACKResponse::~UavvMessageACKResponse(){

}
UavvMessageACKResponse::UavvMessageACKResponse(unsigned char packetID){
    PacketID = packetID;
}

ParseResult UavvMessageACKResponse::TryParse(GimbalPacket packet, UavvMessageACKResponse *msgACKRes){
    if(packet.Data.size() < msgACKRes->Length){
        return ParseResult::InvalidLength;
    }
    msgACKRes->PacketID = packet.Data[0];
    return ParseResult::Success;
}

GimbalPacket UavvMessageACKResponse::Encode(){
    unsigned char data[1];
    data[0] = PacketID;
    return GimbalPacket(UavvGimbalProtocol::MessageAcknowledgement, data, sizeof(data));
}
