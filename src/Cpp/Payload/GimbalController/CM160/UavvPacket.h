#ifndef GIMBALPACKET_H
#define GIMBALPACKET_H
#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include "UavvPacketHelper.h"
#include "UavvGimbalProtocol.h"
#include "Utils/Bytes/ByteManipulation.h"
using namespace std;
enum class ParseResult
{
    Success,
    InvalidLength,
    InvalidData,
    InvalidId,
};
class GimbalPacket
{
public:
    GimbalPacket();
    GimbalPacket(unsigned char idByte, unsigned char *data,int dataLength);
    GimbalPacket(unsigned char idByte, vector<unsigned char> data);
    GimbalPacket(UavvGimbalProtocol id, unsigned char *data,int dataLength);
    GimbalPacket(UavvGimbalProtocol id, vector<unsigned char> data);
    unsigned char IDByte;
    vector<unsigned char> Data;
    unsigned char Sync1 = 0x24;
    unsigned char Sync2 = 0x40;

    vector<unsigned char> encode();
    static unsigned char CalculateChecksum(vector<unsigned char> data, int startIndex, int length);

};

#endif // GIMBALPACKET_H
