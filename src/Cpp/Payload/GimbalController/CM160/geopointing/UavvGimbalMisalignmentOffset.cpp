#include "UavvGimbalMisalignmentOffset.h"


UavvGimbalMisalignmentOffset::UavvGimbalMisalignmentOffset() {
    MountType = 0;
    Pan = 0;
    Tilt = 0;
    for(int i=0; i< 38; i++){
        Reserved[i]=0;
    }
}
UavvGimbalMisalignmentOffset::~UavvGimbalMisalignmentOffset() {}

UavvGimbalMisalignmentOffset::UavvGimbalMisalignmentOffset(unsigned char mountType)
{
    MountType = mountType;
    Pan = 0;
    Tilt = 0;
    for(int i=0; i< 38; i++){
        Reserved[i]=0;
    }
}

ParseResult UavvGimbalMisalignmentOffset::TryParse(GimbalPacket packet, UavvGimbalMisalignmentOffset*GimbalMisalignmentOffset)
{
    if (packet.Data.size() < GimbalMisalignmentOffset->Length)
	{
        return ParseResult::InvalidLength;
	}
    GimbalMisalignmentOffset->MountType = packet.Data[0];
    GimbalMisalignmentOffset->Pan = ByteManipulation::ToUInt16(packet.Data.data(),1,Endianness::Big);
    GimbalMisalignmentOffset->Tilt = ByteManipulation::ToUInt16(packet.Data.data(),3,Endianness::Big);
    for(int i=5; i< GimbalMisalignmentOffset->Length; i++){
        GimbalMisalignmentOffset->Reserved[i-5]=packet.Data[i];
    }
	return ParseResult::Success;
}

GimbalPacket UavvGimbalMisalignmentOffset::Encode()
{
    unsigned char data[43];

    data[0] = MountType;
    ByteManipulation::ToBytes((unsigned short)Pan,Endianness::Big,data,1);
    ByteManipulation::ToBytes((unsigned short)Tilt,Endianness::Big,data,3);
    for(int i=5; i< sizeof(data); i++){
        data[i]=0;
    }
	return GimbalPacket(UavvGimbalProtocol::GimbalMisalignmentOffset, data, sizeof(data));
}
