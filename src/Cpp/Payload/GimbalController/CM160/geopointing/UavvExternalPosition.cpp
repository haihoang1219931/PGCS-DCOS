#include "UavvExternalPosition.h"

UavvExternalPosition::UavvExternalPosition()
{
    //ctor
}

UavvExternalPosition::~UavvExternalPosition()
{
    //dtor
}
ParseResult UavvExternalPosition::TryParse(GimbalPacket packet, UavvExternalPosition *ExternalPosition){
if (packet.Data.size() < ExternalPosition->Length)
	{
        return ParseResult::InvalidLength;
	}
    unsigned flag = 0x00;
    int encodeLat = ByteManipulation::ToInt32(packet.Data.data(),1,Endianness::Little);
    int encodeLon = ByteManipulation::ToInt32(packet.Data.data(),5,Endianness::Little);

    ExternalPosition->Latitude = encodeLat / pow(2,31) * 90.0f;
    ExternalPosition->Longtitude = encodeLon / pow(2,31) * 180.0f;
    ExternalPosition->Altitude = ByteManipulation::ToFloat(packet.Data.data(),9,Endianness::Little);
	return ParseResult::Success;
}
GimbalPacket UavvExternalPosition::Encode(){
    unsigned char data[13];
    data[0] = Flag;
    int encodeLat = (int)(Latitude / 90.0f * pow(2,31));
    int encodeLon = (int)(Longtitude / 180.0f * pow(2,31));
    ByteManipulation::ToBytes(encodeLat,Endianness::Little,data,1);
    ByteManipulation::ToBytes(encodeLon,Endianness::Little,data,5);
    // printf("lat,lon,altitude)=(%f,%f,%f)\r\n",Latitude,Longtitude,Altitude);
    ByteManipulation::ToBytes((int)(Altitude*1000.0f),Endianness::Little,data,9);
	return GimbalPacket(UavvGimbalProtocol::ExternalPosition, data, sizeof(data));
}
