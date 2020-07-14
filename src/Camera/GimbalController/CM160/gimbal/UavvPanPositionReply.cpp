#include "UavvPanPositionReply.h"

UavvPanPositionReply::UavvPanPositionReply(){

}

UavvPanPositionReply::~UavvPanPositionReply(){

}

UavvPanPositionReply::UavvPanPositionReply(float panPosition){
    PanPositionReply = panPosition;
}

ParseResult UavvPanPositionReply::TryParse(GimbalPacket packet, UavvPanPositionReply *setPanTiltPositionPacket){
    if(packet.Data.size() < setPanTiltPositionPacket->Length){
        return ParseResult::InvalidLength;
    }
    float panPosition;
    panPosition = UavvPacketHelper::PacketToAngle(packet.Data[0], packet.Data[1]);
    setPanTiltPositionPacket->PanPositionReply = panPosition;
    return ParseResult::Success;
}

GimbalPacket UavvPanPositionReply::Encode(){
    unsigned char data[2];
    vector<unsigned char>tmp_PanPosition;
    tmp_PanPosition = UavvPacketHelper::PositionToPacket(PanPositionReply);

    for (int i = 0; i < 2; i++)
    {
        data[i] = tmp_PanPosition[i];
    }

    return GimbalPacket(UavvGimbalProtocol::SetPanPosition, data, sizeof(data));
}
