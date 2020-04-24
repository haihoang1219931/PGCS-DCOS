#include"UavEnableEOSensor.h"

UavvEnableEOSensor::UavvEnableEOSensor()
{

}
UavvEnableEOSensor::UavvEnableEOSensor(bool enable){
    if (enable) {
        Status = SensorStatus::Enable;
    }else{
        Status = SensorStatus::Disable;
    }
}

UavvEnableEOSensor::UavvEnableEOSensor(SensorStatus status)
{
    Status = status;
}

UavvEnableEOSensor::~UavvEnableEOSensor(){}

GimbalPacket UavvEnableEOSensor::Encode()
{
	unsigned char data[1];
    if (Status == SensorStatus::Enable)
	{
		data[0] = 1;
	}
	else
	{
		data[0] = 0;
	}
    return GimbalPacket(UavvGimbalProtocol::SensorEnable, data, sizeof(data));
}

ParseResult UavvEnableEOSensor::TryParse(GimbalPacket packet, UavvEnableEOSensor *EnableEOSensor)
{
    if (packet.Data.size() < EnableEOSensor->Length)
	{
        return ParseResult::InvalidLength;
	}
    SensorStatus enableSensorStatus;
	if (packet.Data[0] == 0x00)
	{
        enableSensorStatus = SensorStatus::Disable;
	}
	else if (packet.Data[0] == 0x01)
	{
        enableSensorStatus = SensorStatus::Enable;
	}
	else
        return ParseResult::InvalidData;
	*EnableEOSensor = UavvEnableEOSensor(enableSensorStatus);
    return ParseResult::Success;
}
