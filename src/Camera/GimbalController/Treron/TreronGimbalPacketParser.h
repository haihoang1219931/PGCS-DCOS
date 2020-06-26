#ifndef TRERONGIMBALPACKETPARSER_H
#define TRERONGIMBALPACKETPARSER_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <QObject>
#include "Packet/KLV.h"
using namespace std;
class TreronGimbalPacketParser: public QObject
{
    Q_OBJECT
public:
    TreronGimbalPacketParser(QObject* parent=nullptr);
    ~TreronGimbalPacketParser();
    enum SyncBytes : unsigned char
    {
        Sync1 = 0x24,
        Sync2 = 0x40,
    };
    const int MINIMUM_UAVV_PACKET_SIZE = 7;
    vector<unsigned char> receivedBuffer;
    void Push(vector<unsigned char> data);
    void Push(unsigned char* data, int length);
    void Parse();
    void Reset();
Q_SIGNALS:
    void gimbalPacketParsed(key_type key,vector<byte> data);
};


#endif // TRERONGIMBALPACKETPARSER_H
