#ifndef GIMBALPACKETPARSER_H
#define GIMBALPACKETPARSER_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <QObject>
#include <QQuickItem>
#include "Packet/KLV.h"

using namespace std;
class GimbalPacketParser: public QObject
{
    Q_OBJECT
public:
    GimbalPacketParser(QObject* parent=0);
    virtual ~GimbalPacketParser();
    enum SyncBytes : unsigned char
    {
        Sync1 = 0x24,
        Sync2 = 0x40,
    };
    const int MINIMUM_UAVV_PACKET_SIZE = 7;
    vector<unsigned char> receivedBuffer;
    virtual void Push(vector<unsigned char> data);
    virtual void Push(unsigned char* data, int length);
    virtual void Parse();
    void Reset();
Q_SIGNALS:
    void UavvGimbalPacketParsed(key_type key,vector<byte> data);
};

#endif // GIMBALPACKETPARSER_H
