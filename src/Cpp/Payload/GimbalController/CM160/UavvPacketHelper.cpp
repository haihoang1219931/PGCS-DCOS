#include "UavvPacketHelper.h"
#include <cmath>

UavvPacketHelper::UavvPacketHelper(){}
UavvPacketHelper::~UavvPacketHelper(){}

short int UavvPacketHelper::EncodeEulerAngle(float _angle)
{
    float angle = _angle > 180 ? 180 : _angle;
	angle = angle < -180 ? -180 : angle;
    float a = (angle / 360.0) * 65535.0;
	a = a > 32767 ? 32767 : a;
	return (short int)a;
}

unsigned short int UavvPacketHelper::EncodeYawAngle(float _angle)
{
    float angle = _angle > 360 ? 360 : _angle;
	angle = angle < 0 ? 0 : angle;
	unsigned short int result= (unsigned short int)(angle / 360.0) * 65535.0;
	return result;
}

float UavvPacketHelper::DecodeEulerAngle(short int value)
{
    float result = (float)((float)value / 65535.0) * 360.0;
	return result;
}

float UavvPacketHelper::DecodeYawAngle(unsigned short int value)
{
    float result = (float)((float)value / 65535.0) * 360.0;
	return result;
}

float UavvPacketHelper::PacketToAngle(unsigned char msb, unsigned char lsb)
{
    float result=0;
    result = (float)msb*256.0;
    result = (float)result + lsb;
    result = (float)(result / 32768)*360.0;
	return result;
}

float UavvPacketHelper::PacketToVelocity(unsigned char msb, unsigned char lsb)
{
    float result;
    result = (float)msb*256.0;
    result = (float)result + lsb;
    result = (float)(result - 32768.0)/100.0;
	return result;
}

float UavvPacketHelper::PacketToTemperature(unsigned char msb, unsigned char lsb)
{
    float result;
    result = (float)msb*256.0;
    result = (float)result + lsb;
    result = (float)(result - 32768.0) / 100.0;
	return result;
}

vector<unsigned char>UavvPacketHelper::VelocityToPacket(float _velocity)
{
    float velocity = _velocity > 327 ? 327 : _velocity;
	velocity = velocity < -327 ? -327 : velocity;
    vector<unsigned char> result;
	unsigned short int tmp = (unsigned short int)(velocity * 100.0) + 32768;
    result.push_back((unsigned char)(tmp / 256));
    result.push_back((unsigned char)(tmp % 256));
	return result;
}

vector<unsigned char>UavvPacketHelper::PositionToPacket(float _angle)
{
    float angle = (int)_angle %360;
	if (angle < 0)
	{
		angle = 360 + angle;
	}
	int data = (int)(angle / 360 * 32768);
    vector<unsigned char> result = convert_to_databytes(data);
	return result;
}

vector<unsigned char>UavvPacketHelper::convert_to_databytes(int data)
{
    vector<unsigned char> temp;
    temp.push_back((unsigned char)floor(data / 256.0));
    temp.push_back((unsigned char)(data % 256));
	return temp;
}

ControlModeType UavvPacketHelper::ByteToControlMode(unsigned char value)
{
	unsigned char result;
	for (int i = 0; i < 7; i++)
	{
		if (value == i) result = value;
		return (ControlModeType)result;
	}
    return ControlModeType::Undefined;
}

unsigned char UavvPacketHelper::CalculateChecksum(vector<unsigned char> data, int offset, int length)
{
	int sum=0;
    for (int i = offset; i < length + offset; i++)
	{
		sum += data[i];
	}
	return (255 - (sum % 255));
}

unsigned char UavvPacketHelper::CalculateChecksum(vector<unsigned char> data)
{
    int sum=0;
    for (unsigned int i = 0; i < data.size(); i++)
    {
        sum += data[i];
    }
    return (255 - (sum % 255));
}


float UavvPacketHelper::PacketToLatitude(vector<unsigned char> data)
{
    int tmp = ByteManipulation::ToInt32(data.data(), 0, Endianness::Big);
	return (tmp * 90.0f / 2.147484E+09);
}

float UavvPacketHelper::PacketToLongitude(vector<unsigned char> data)
{
    int tmp = ByteManipulation::ToInt32(data.data(), 0, Endianness::Big);
	return (tmp * 180.0f / 2.147484E+09);
}

const double LONGITUDE2INT_CONVERSION = 2147483647.0 / 180.0; //2^31-1/180
const double LATITUDE2INT_CONVERSION = 2147483647.0 / 90.0; //2^31-1/180

vector<unsigned char>UavvPacketHelper::LongitudeToPacket(float longitude)
{
	if (longitude >= -180 && longitude <= 180)
	{
		int tmp = (int)longitude * LONGITUDE2INT_CONVERSION;
        return ByteManipulation::ToBytes(tmp, Endianness::Big);
	}
	else
	{
        return ByteManipulation::ToBytes((int)(-2147483647 - 1), Endianness::Big);
	}
}

vector<unsigned char>UavvPacketHelper::LatitudeToPacket(float latitude)
{
	if (latitude >= -90 && latitude <= 90)
	{

		int tmp = (int)(latitude * LATITUDE2INT_CONVERSION);
        return ByteManipulation::ToBytes(tmp, Endianness::Big);
	}
	else
	{
        return ByteManipulation::ToBytes((int)(-2147483647 - 1), Endianness::Big);
	}
}

float UavvPacketHelper::PacketToLatitude(int encodedValue)
{
	float latitude = (float)encodedValue / LATITUDE2INT_CONVERSION;
	return latitude;
}

float UavvPacketHelper::PacketToLongitude(int encodedValue)
{
	float longitude = (float)encodedValue / LONGITUDE2INT_CONVERSION;
	return longitude;
}



