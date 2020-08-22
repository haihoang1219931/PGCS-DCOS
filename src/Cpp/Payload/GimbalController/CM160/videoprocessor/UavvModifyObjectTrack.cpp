#include "UavvModifyObjectTrack.h"


UavvModifyObjectTrack::UavvModifyObjectTrack() {}
UavvModifyObjectTrack::~UavvModifyObjectTrack() {}

UavvModifyObjectTrack::UavvModifyObjectTrack(ActionType action, unsigned int column, unsigned int row)
{
	setActModifyObjectTrack(action);
	setColumnModifyObjectTrack(column);
	setRowModifyObjectTrack(row);
}

ParseResult UavvModifyObjectTrack::TryParse(GimbalPacket packet, UavvModifyObjectTrack *ModifyObjectTrack)
{
    if (packet.Data.size() < ModifyObjectTrack->Length)
	{
        return ParseResult::InvalidLength;
	}
    ActionType action;
    printf("track packet.Data[0] = %02X\r\n",packet.Data[0]);
	if (packet.Data[0] == 0x80)
        action = ActionType::StartGimbalTracking;
	else if (packet.Data[0] == 0x00)
        action = ActionType::AddTrackCoordinates;
	else if (packet.Data[0] == 0x01)
        action = ActionType::AddTrackCrosshair;
	else if (packet.Data[0] == 0x02)
        action = ActionType::DesignatesTrack;
	else if (packet.Data[0] == 0x03)
        action = ActionType::AddTrackPrimary;
	else if (packet.Data[0] == 0x04)
        action = ActionType::StopGimbalTrack;
	else if (packet.Data[0] == 0x05)
        action = ActionType::KillTrackNearest;
	else if (packet.Data[0] == 0x06)
        action = ActionType::KillAllTracks;
	else if (packet.Data[0] == 0x07)
        action = ActionType::KillAllPrimary;
	else if (packet.Data[0] == 0x08)
        action = ActionType::KillAllCoordinates;
	else if (packet.Data[0] == 0x09)
        action = ActionType::KillAllCrosshair;
	else
        return ParseResult::InvalidData;
    //column = (packet.Data[1] << 8) | (packet.Data[2]);
    //row = (packet.Data[3] << 8) | (packet.Data[4]);
    ModifyObjectTrack->Column = ByteManipulation::ToUInt16(packet.Data.data(),1,Endianness::Big);
    ModifyObjectTrack->Row = ByteManipulation::ToUInt16(packet.Data.data(),3,Endianness::Big);
    ModifyObjectTrack->action = action;
	return ParseResult::Success;
}

GimbalPacket UavvModifyObjectTrack::Encode()
{
	unsigned char data[5];
    data[0] = 0x88;
    if (getActModifyObjectTrack() == ActionType::AddTrackCoordinates)
        data[0] |= 0x00 & 0x7F;
    else if (getActModifyObjectTrack() == ActionType::AddTrackCrosshair)
        data[0] |= 0x01 & 0x7F;
    else if (getActModifyObjectTrack() == ActionType::DesignatesTrack)
        data[0] |= 0x02 & 0x7F;
    else if (getActModifyObjectTrack() == ActionType::AddTrackPrimary)
        data[0] |= 0x03 & 0x7F;
    else if (getActModifyObjectTrack() == ActionType::StopGimbalTrack)
        data[0] |= 0x04 & 0x7F;
    else if (getActModifyObjectTrack() == ActionType::KillTrackNearest)
        data[0] |= 0x05 & 0x7F;
    else if (getActModifyObjectTrack() == ActionType::KillAllTracks){
        data[0] |= 0x00 & 0x7F;
        printf("KillAllTracks\r\n");
    }else if (getActModifyObjectTrack() == ActionType::KillAllPrimary)
        data[0] |= 0x07 & 0x7F;
    else if (getActModifyObjectTrack() == ActionType::KillAllCoordinates)
        data[0] |= 0x08 & 0x7F;
    else if (getActModifyObjectTrack() == ActionType::KillAllCrosshair)
        data[0] |= 0x09 & 0x7F;
    ByteManipulation::ToBytes(Column,Endianness::Little,data,1);
    ByteManipulation::ToBytes(Row,Endianness::Little,data,3);
	return GimbalPacket(UavvGimbalProtocol::ModifyObjectTrack, data, sizeof(data));
}
