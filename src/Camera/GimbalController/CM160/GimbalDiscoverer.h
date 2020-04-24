#ifndef GIMBALDISCOVERER_H
#define GIMBALDISCOVERER_H

#include <stdio.h>
#include <iostream>
#include <QQuickItem>
#include <QObject>
#include <QHostAddress>
#include "UDPSenderListener.h"
#include "Bytes/ByteManipulation.h"
using namespace std;
class DiscoverPacket{
public:
    int PacketLength = 104;
    int ID;
    int Length;
    int MinorVersion;
    int MajorVersion;
    int SoftwareFeatures;
    int HardwareType;
    string MacAddress;
    string IPAddress;
    string VideoAddress;
    string Name;
    int VideoPort;
    int ComsPort;
    DiscoverPacket(){}
    ~DiscoverPacket(){}
    DiscoverPacket(int id, int length,
                   int minorVersion, int majorVersion,
                   int softwarefeatures, int hardwaretype,
                   string macAddress,
                   string ipAddress,
                   string videoAdress,
                   string name,
                   int videoPort, int comsPort){
        ID = id;
        Length = length;
        MinorVersion = minorVersion;
        MajorVersion = majorVersion;
        SoftwareFeatures = softwarefeatures;
        HardwareType = hardwaretype;
        MacAddress = macAddress;
        IPAddress = ipAddress;
        VideoAddress = videoAdress;
        Name = name;
        VideoPort = videoPort;
        ComsPort = comsPort;
    }
    static bool TryParse(unsigned char* packet,int packetLength, DiscoverPacket *discoverPacket){
        if (packetLength < 104){
            return false;
        }

        //return true;
        int id=0, length=0, videoPort=0, comsPort=0;
        ushort minorVersion=0, majorVersion=0, softwareFeartures=0, hardwareType=0;
        char mac[20], ipAddress[16], videoAddress[16], name[32];
        id = ByteManipulation::ToInt32(packet, 0, Endianness::Little);
        length = ByteManipulation::ToInt32(packet, 4, Endianness::Little);
        minorVersion = ByteManipulation::ToUInt16(packet, 8, Endianness::Little);
        majorVersion = ByteManipulation::ToUInt16(packet, 10, Endianness::Little);
        softwareFeartures = ByteManipulation::ToUInt16(packet, 12, Endianness::Little);
        hardwareType = ByteManipulation::ToUInt16(packet, 14, Endianness::Little);
        for (int j = 0; j<20; j++)
        {
            mac[j] = packet[16 + j];
        }
        for (int j = 0; j < 16; j++)
        {
            ipAddress[j] = packet[36 + j];
        }
        for (int j = 0; j < 16; j++)
        {
            videoAddress[j] = packet[52 + j];
        }
        for (int j = 0; j < 32; j++)
        {
            name[j] = packet[68 + j];
        }
        videoPort = ByteManipulation::ToUInt16(packet, 100, Endianness::Little);
        comsPort = ByteManipulation::ToUInt16(packet, 102, Endianness::Little);
        comsPort = ByteManipulation::ToUInt16(packet, 102, Endianness::Little);
        discoverPacket->ID = id;
        qDebug () << "id: "<<id;
        discoverPacket->Length = length;
        qDebug () << "length: "<<length;
        discoverPacket->MinorVersion = minorVersion;
        qDebug () << "minorVersion: "<<minorVersion;
        discoverPacket->MajorVersion = majorVersion;
        qDebug () << "majorVersion: "<<majorVersion;
        discoverPacket->SoftwareFeatures = softwareFeartures;
        qDebug () << "softwareFeartures: "<<softwareFeartures;
        discoverPacket->HardwareType = hardwareType;
        qDebug () << "hardwareType: "<< hardwareType;

        string _ipAddress(ipAddress);
        qDebug ("ipAddress: %s",ipAddress);
        discoverPacket->IPAddress = ipAddress;

        string _mac(mac);
        qDebug ("mac: %s",mac);
        discoverPacket->MacAddress = _mac;

        string _videoAddress(videoAddress);
        qDebug ("videoAddress: %s",videoAddress);
        discoverPacket->VideoAddress = _videoAddress;

        string _name(name);
        qDebug ("name: %s",name);
        discoverPacket->Name = _name;
        discoverPacket->VideoPort = videoPort;
        qDebug () << "videoPort: "<< videoPort;
        discoverPacket->ComsPort = comsPort;
        qDebug () << "comsPort: "<< comsPort;
        qDebug () << "Parse discorver packet done";
        return true;
    }
    void Package(unsigned char *packett,int *packetLength, int id, int length, int minorVersion, int majorVersion, int softwarefeatures, int hardwaretype, char* macAddress, char* ipAddress, char* videoAddress, char* name, int videoPort, int comsPort){
        /*
        *packetLength = 104;
        if(packett == NULL){
            packett = new unsigned char[*packetLength];
        }
        packett = (ByteManipulation::ToBytes(id, ByteManipulation::Little));

        *(packett + 4 ) = (unsigned char)(ByteManipulation::ToBytes(length, ByteManipulation::Little));

        *(packett + 8 ) = (unsigned char)(ByteManipulation::ToBytes(minorVersion, ByteManipulation::Little));

        *(packett + 10) = (unsigned char)(ByteManipulation::ToBytes(majorVersion, ByteManipulation::Little));

        *(packett + 12) = (unsigned char)(ByteManipulation::ToBytes(softwarefeatures, ByteManipulation::Little));

        *(packett + 14) = (unsigned char)(ByteManipulation::ToBytes(hardwaretype, ByteManipulation::Little));

        for (int j = 0; j < 20; j++)
        {
            if (*(macAddress) != '\0')
                *(packett + 16 + j) = *(macAddress + j);
            else
                *(packett + 16 + j) = ' ';
        }

        for (int j = 0; j < 16; j++)
        {
            if (*(ipAddress+j) != '\0')
                *(packett + 36 + j) = *(ipAddress + j);
            else
                *(packett + 36 + j) = ' ';
        }
        for (int j = 0; j < 16; j++)
        {
            if (*(videoAddress+j) != '\0')
                *(packett + 52 + j) = *(videoAddress + j);
            else
                *(packett + 52 + j) = ' ';
        }

        for (int j = 0; j < 32; j++)
        {
            if (*(name+j) != '\0')
                *(packett + 68 + j) = *(name + j);
            else
                *(packett + 68 + j) = ' ';
        }
        *(packett + 100) = (unsigned char)(ByteManipulation::ToBytes(videoPort, ByteManipulation::Little));
        *(packett + 102) = (unsigned char)(ByteManipulation::ToBytes(comsPort, ByteManipulation::Little));
        */
    }
};

class GimbalDiscoverer: public QObject
{
    Q_OBJECT
public:
    vector<DiscoverPacket> lstGimbal;
    GimbalDiscoverer(QObject* parent = 0);
    virtual ~GimbalDiscoverer();
    UdpSenderListener *socket;
    const int UAVV_DISCOVER_PORT = 50000;
    const QString discoverMessage = "SLDISCOVER";
    Q_INVOKABLE void requestDiscover();
    Q_INVOKABLE QVariantMap GetListGimbal();
    void Dispose();
Q_SIGNALS:
    void newDiscoverPacketReceived(QString name,QString ip,
                                   QString videoIP,int videoPort,
                                   int sendPort,int recvPort);
public Q_SLOTS:
    void handleNewMessageRecieved();
};

#endif // GIMBALDISCOVERER_H
