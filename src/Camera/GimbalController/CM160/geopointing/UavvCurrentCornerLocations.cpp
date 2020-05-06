#include "UavvCurrentCornerLocations.h"

UavvCurrentCornerLocation::UavvCurrentCornerLocation() {}
UavvCurrentCornerLocation::~UavvCurrentCornerLocation() {}

UavvCurrentCornerLocation::UavvCurrentCornerLocation(float tlLatitude,
                                                     float tlLongitude,
                                                     float trLatitude,
                                                     float trLongitude,
                                                     float brLatitude,
                                                     float brLongitude,
                                                     float blLatitude,
                                                     float blLongitude)
{

}

ParseResult UavvCurrentCornerLocation::TryParse(GimbalPacket packet, UavvCurrentCornerLocation*CurrentCornerLocation)
{
    if (packet.Data.size() < CurrentCornerLocation->Length)
	{
        return ParseResult::InvalidLength;
	}
    int tlLat,tlLon;
    int trLat,trLon;
    int brLat,brLon;
    int blLat,blLon;
    CurrentCornerLocation->Reserved01 = ByteManipulation::ToInt16(packet.Data.data(),0,Endianness::Big);
    tlLat = ByteManipulation::ToInt32(packet.Data.data(),2,Endianness::Big);
    tlLon = ByteManipulation::ToInt32(packet.Data.data(),6,Endianness::Big);
    CurrentCornerLocation->TopLeftLatitude = (float)((double)((double)tlLat + 1 -pow(2,31))  * 180 / pow(2,32))+90;
    CurrentCornerLocation->TopLeftLongitude = (float)((double)((double)tlLon + 1 -pow(2,31))  * 360 / pow(2,32))+180;

    CurrentCornerLocation->Reserved02 = ByteManipulation::ToInt16(packet.Data.data(),10,Endianness::Big);
    trLat = ByteManipulation::ToInt32(packet.Data.data(),12,Endianness::Big);
    trLon = ByteManipulation::ToInt32(packet.Data.data(),16,Endianness::Big);
    CurrentCornerLocation->TopRightLatitude = (float)((double)((double)trLat + 1 -pow(2,31))  * 180 / pow(2,32))+90;
    CurrentCornerLocation->TopRightLongitude = (float)((double)((double)trLon + 1 -pow(2,31))  * 360 / pow(2,32))+180;

    CurrentCornerLocation->Reserved03 = ByteManipulation::ToInt16(packet.Data.data(),20,Endianness::Big);
    brLat = ByteManipulation::ToInt32(packet.Data.data(),22,Endianness::Big);
    brLon = ByteManipulation::ToInt32(packet.Data.data(),26,Endianness::Big);
    CurrentCornerLocation->BottomRightLatitude = (float)((double)((double)brLat + 1 -pow(2,31))  * 180 / pow(2,32))+90;
    CurrentCornerLocation->BottomRightLongitude = (float)((double)((double)brLon + 1 -pow(2,31))  * 360 / pow(2,32))+180;

    CurrentCornerLocation->Reserved04 = ByteManipulation::ToInt16(packet.Data.data(),30,Endianness::Big);
    blLat = ByteManipulation::ToInt32(packet.Data.data(),32,Endianness::Big);
    blLon = ByteManipulation::ToInt32(packet.Data.data(),36,Endianness::Big);
    CurrentCornerLocation->BottomLeftLatitude = (float)((double)((double)blLat + 1 -pow(2,31))  * 180 / pow(2,32))+90;
    CurrentCornerLocation->BottomLeftLongitude = (float)((double)((double)blLon + 1 -pow(2,31))  * 360 / pow(2,32))+180;

    CurrentCornerLocation->Reserved05 = ByteManipulation::ToInt16(packet.Data.data(),40,Endianness::Big);
    CurrentCornerLocation->CenterLatitude = (CurrentCornerLocation->TopLeftLatitude +
                                             CurrentCornerLocation->TopRightLatitude+
                                             CurrentCornerLocation->BottomRightLatitude+
                                             CurrentCornerLocation->BottomLeftLatitude) / 4.0f;
    CurrentCornerLocation->CenterLongitude = (CurrentCornerLocation->TopLeftLongitude +
                                             CurrentCornerLocation->TopRightLongitude+
                                             CurrentCornerLocation->BottomRightLongitude+
                                             CurrentCornerLocation->BottomLeftLongitude) / 4.0f;
    return ParseResult::Success;
}

GimbalPacket UavvCurrentCornerLocation::Encode()
{
	unsigned char data[42];
    ByteManipulation::ToBytes(Reserved01,Endianness::Big,data,0);
    ByteManipulation::ToBytes(TopLeftLatitude,Endianness::Big,data,2);
    ByteManipulation::ToBytes(TopLeftLongitude,Endianness::Big,data,6);

    ByteManipulation::ToBytes(Reserved02,Endianness::Big,data,10);
    ByteManipulation::ToBytes(TopRightLatitude,Endianness::Big,data,12);
    ByteManipulation::ToBytes(TopRightLongitude,Endianness::Big,data,16);

    ByteManipulation::ToBytes(Reserved03,Endianness::Big,data,20);
    ByteManipulation::ToBytes(BottomRightLatitude,Endianness::Big,data,22);
    ByteManipulation::ToBytes(BottomRightLongitude,Endianness::Big,data,26);

    ByteManipulation::ToBytes(Reserved04,Endianness::Big,data,30);
    ByteManipulation::ToBytes(BottomLeftLatitude,Endianness::Big,data,32);
    ByteManipulation::ToBytes(BottomLeftLatitude,Endianness::Big,data,36);

    ByteManipulation::ToBytes(Reserved05,Endianness::Big,data,40);
	return GimbalPacket(UavvGimbalProtocol::CurrentCornerLocations, data, sizeof(data));
}
