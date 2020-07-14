#ifndef CM160GIMBALPACKETPARSER_H
#define CM160GIMBALPACKETPARSER_H

#include <stdio.h>
#include <iostream>
#include <vector>
#include <QObject>
#include <QQuickItem>
#include "UavvPacket.h"
using namespace std;
class CM160GimbalPacketParser: public QObject
{
    Q_OBJECT
public:
    CM160GimbalPacketParser(QObject* parent=0);
    ~CM160GimbalPacketParser();
    enum SyncBytes : unsigned char
    {
        Sync1 = 0x24,
        Sync2 = 0x40,
    };
    const int MINIMUM_UAVV_PACKET_SIZE = 5;
    vector<unsigned char> receivedBuffer;
    void Push(vector<unsigned char> data);
    void Push(unsigned char* data, int length);
    void Parse();
    void Reset();
Q_SIGNALS:
    void gimbalPacketParsed(GimbalPacket packet, unsigned char checksum);
};

#endif // CM160GIMBALPACKETPARSER_H
