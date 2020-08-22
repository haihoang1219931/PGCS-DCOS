#ifndef UAVVVIDEODESTINATION_H
#define UAVVVIDEODESTINATION_H

#include "../UavvPacket.h"
class UavvVideoDestination
{
public:
    unsigned int Length = 8;
    unsigned int IPAddress;
    unsigned short Port;
	unsigned char NetworkStream, AnalogOut;

    void setIPAdressVideoDestination(unsigned int address)
	{
		IPAddress = address;
	}

    void setPortVideoDestination(unsigned short port)
	{
		Port = port;
	}

	void setNerworkStreamVideoDestination(unsigned char networkStream)
	{
		NetworkStream = networkStream;
	}

	void setAnalogOutDestination(unsigned char analogOut)
	{
		AnalogOut = analogOut;
	}

    unsigned int getIPAdressVideoDestination()
	{
		return IPAddress;
	}

    unsigned short getPortVideoDestination()
	{
		return Port;
	}

	unsigned char getNetworkStreamVideoDestination()
	{
		return NetworkStream;
	}

	unsigned char getAnalogOutVideoDestination()
	{
		return AnalogOut;
	}

	UavvVideoDestination();
	~UavvVideoDestination();
    UavvVideoDestination(unsigned int address, unsigned short port, unsigned char networkStream, unsigned char analogOut);
	static ParseResult TryParse(GimbalPacket packet, UavvVideoDestination *VideoDestination);
    GimbalPacket Encode();
};
#endif
