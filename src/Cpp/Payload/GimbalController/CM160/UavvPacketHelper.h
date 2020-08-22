#ifndef UAVVPACKETHELPER_H
#define UAVVPACKETHELPER_H
#include "UavvGimbalProtocol.h"
#include "Utils/Bytes/ByteManipulation.h"
class UavvPacketHelper
{
public :
	UavvPacketHelper();
	~UavvPacketHelper();
    static short int EncodeEulerAngle(float angle);
    static unsigned short int EncodeYawAngle(float angle);
    static float DecodeEulerAngle(short int value);
    static float DecodeYawAngle(unsigned short int value);
    static float PacketToAngle(unsigned char msb, unsigned char lsb);
    static float PacketToVelocity(unsigned char msb, unsigned char lsb);
    static float PacketToTemperature(unsigned char msb, unsigned char lsb);
    static vector<unsigned char>VelocityToPacket(float velocity);
    static vector<unsigned char>PositionToPacket(float angle);
	static ControlModeType ByteToControlMode(unsigned char value);
    static unsigned char CalculateChecksum(vector<unsigned char> data, int offset, int length);
    static unsigned char CalculateChecksum(vector<unsigned char> data);
    static float PacketToLatitude(vector<unsigned char> data);
    static float PacketToLongitude(vector<unsigned char> data);

    static vector<unsigned char>LongitudeToPacket(float longitude);
    static vector<unsigned char>LatitudeToPacket(float latitude);
	static float PacketToLatitude(int encodedValue);
	static float PacketToLongitude(int encodedValue);
    static vector<unsigned char>convert_to_databytes(int data);
};
#endif
