#include"UavSensorCurrentFoV.h"

UavvSensorCurrentFoV::UavvSensorCurrentFoV(){}

UavvSensorCurrentFoV::UavvSensorCurrentFoV(ImageSensorType type, float hfov, float vfov)
{
//    Type = type;
//    Horizontal = hfov;
//    Vertical = vfov;
}

UavvSensorCurrentFoV::~UavvSensorCurrentFoV(){}

GimbalPacket UavvSensorCurrentFoV::Encode()
{
	unsigned char data[5];
//    if (Type == ImageSensorType::EOSensor)
//		data[0] = 0;
//	else
//		data[0] = 1;
//    unsigned short _hfov = Horizontal/180.0f * 16384.0;
//    unsigned short _vfov = Vertical/180.0f * 16384.0;
//    ByteManipulation::ToBytes(_hfov,Endianness::Big, data,1);
//    ByteManipulation::ToBytes(_vfov,Endianness::Big, data,3);
    return GimbalPacket(UavvGimbalProtocol::SensorFieldOfView, data, sizeof(data));
}

ParseResult UavvSensorCurrentFoV::TryParse(GimbalPacket packet, UavvSensorCurrentFoV *SensorCurrentFoV)
{

    if (packet.Data.size() % 5 != 0)
	{
        printf("Invalid length\r\n");
        return ParseResult::InvalidLength;
	}
//    printf("Data: ");
//    for(int i=0; i< packet.Data.size(); i++){
//        printf(" %02X",packet.Data[i]);
//    }
//    printf("\r\n");
    SensorCurrentFoV->numSensor = packet.Data.size() / 5;
    for(int i=0 ; i< SensorCurrentFoV->numSensor; i++){
        ImageSensorType _Type;
        unsigned short _hfov, _vfov;
        if (packet.Data[0+i*5] == 0x00){
            _Type = ImageSensorType::EOSensor;
//            printf("EO sensor: ");
        }else if (packet.Data[0+i*5] == 0x01){
            _Type = ImageSensorType::IRSensor;
//            printf("IR sensor: ");
        }else{
//            printf("Invalid data\r\n");
            return ParseResult::InvalidData;
        }

        _hfov = ByteManipulation::ToUInt16(packet.Data.data(),1+i*5,Endianness::Big);
        _vfov = ByteManipulation::ToUInt16(packet.Data.data(),3+i*5,Endianness::Big);
//        printf("[%f,%f]\r\n",(float)_hfov/16348.0*180.0,(float)_vfov/16348.0*180.0);
        SensorCurrentFoV->Type.push_back(_Type);
        SensorCurrentFoV->Horizontal.push_back((float)_hfov/16348.0*180.0);
        SensorCurrentFoV->Vertical.push_back((float)_vfov/16348.0*180.0);
    }

    return ParseResult::Success;
}
