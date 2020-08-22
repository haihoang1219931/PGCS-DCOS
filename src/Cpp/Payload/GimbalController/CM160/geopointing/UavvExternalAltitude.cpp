#include "UavvExternalAltitude.h"
//#include "../externalpacket.hpp"
UavvExternalAltitude::UavvExternalAltitude()
{
    //ctor
}

UavvExternalAltitude::~UavvExternalAltitude()
{
    //dtor
}
ParseResult UavvExternalAltitude::TryParse(GimbalPacket packet, UavvExternalAltitude *ExternalAltitude){
    if (packet.Data.size() < ExternalAltitude->Length)
	{
        return ParseResult::InvalidLength;
	}
    unsigned flag = packet.Data[0];
    ExternalAltitude->UseRoll = (flag&0x01)!=0?true:false;
    ExternalAltitude->UsePitch = (flag&0x02)!=0?true:false;
    ExternalAltitude->UseYaw = (flag&0x04)!=0?true:false;
    ExternalAltitude->SaveRoll = (flag&0x08)!=0?true:false;
    ExternalAltitude->SavePitch = (flag&0x10)!=0?true:false;
    ExternalAltitude->SaveYaw = (flag&0x20)!=0?true:false;

    float roll  = (float)ByteManipulation::ToInt32(packet.Data.data(),1,Endianness::Big);
    ExternalAltitude->Roll = roll*180.0f/2147483647.0f ;
    float pitch = (float)ByteManipulation::ToInt32(packet.Data.data(),5,Endianness::Big);
    ExternalAltitude->Pitch = pitch *180.0f/2147483647.0f ;
    float yaw = (float)ByteManipulation::ToUInt32(packet.Data.data(),9,Endianness::Big);
    ExternalAltitude->Yaw = yaw *360.0f/4294967294.0f;
	return ParseResult::Success;
}
GimbalPacket UavvExternalAltitude::Encode(){
    unsigned char data[13];
    unsigned char flag = 0x00;
    flag |= UavvExternalAltitude::UseRoll==true?0x01:0x00;
    flag |= UavvExternalAltitude::UsePitch==true?0x02:0x00;
    flag |= UavvExternalAltitude::UseYaw==true?0x04:0x00;
    flag |= UavvExternalAltitude::SaveRoll==true?0x08:0x00;
    flag |= UavvExternalAltitude::SavePitch==true?0x10:0x00;
    flag |= UavvExternalAltitude::SaveYaw==true?0x20:0x00;
    int roll = (int)(UavvExternalAltitude::Roll * 2147483647.0f / 180.0f);
    int pitch = (int)(UavvExternalAltitude::Pitch  * 2147483647.0f / 180.0f);
    unsigned int yaw = (unsigned int)(UavvExternalAltitude::Yaw  * 4294967294.0f / 360.0f);
    data[0]=flag;
    ByteManipulation::ToBytes(roll,Endianness::Little,data,1);
    ByteManipulation::ToBytes(pitch,Endianness::Little,data,5);
    ByteManipulation::ToBytes(yaw,Endianness::Little,data,9);
	return GimbalPacket(UavvGimbalProtocol::ExternalAltitude, data, sizeof(data));
}
