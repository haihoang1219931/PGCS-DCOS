#include "UavvTiltPositionReply.h"

UavvTiltPositionReply::UavvTiltPositionReply(){

}

UavvTiltPositionReply::~UavvTiltPositionReply(){

}

UavvTiltPositionReply::UavvTiltPositionReply(float tiltVelocity){
    TiltVelocity = tiltVelocity;
}

ParseResult UavvTiltPositionReply::TryParse(GimbalPacket packet, UavvTiltPositionReply *tiltVelocityPacket){
    if(packet.Data.size() < tiltVelocityPacket->Length){
        return ParseResult::InvalidLength;
    }
    float tiltVelocity;
    tiltVelocity = UavvPacketHelper::PacketToAngle(packet.Data[0], packet.Data[1]);
    tiltVelocityPacket->TiltVelocity = tiltVelocity;
    return ParseResult::Success;
}

GimbalPacket UavvTiltPositionReply::Encode(){
    unsigned char data[2];
    vector<unsigned char>tmp_PanPosition;
    tmp_PanPosition = UavvPacketHelper::PositionToPacket(TiltVelocity);

    for (int i = 0; i < 2; i++)
    {
        data[i] = tmp_PanPosition[i];
    }

    return GimbalPacket(UavvGimbalProtocol::SetPanPosition, data, sizeof(data));
}
