#ifndef UDPMULTICASTLINK_H
#define UDPMULTICASTLINK_H

#include <QString>
#include <QUdpSocket>
#include <QDataStream>
#include <QHostAddress>
#include <QUdpSocket>
#include <QtNetwork>
#include <QTimer>
#include <vector>
#include <stdio.h>
#include <iostream>
#include "LinkInterface.h"
using namespace std;
class UDPMulticastLink : public LinkInterface
{
    Q_OBJECT
public:
    explicit UDPMulticastLink(LinkInterface *parent = nullptr);
    virtual ~UDPMulticastLink();

    bool isOpen() override;
    void loadConfig(Config* config) override;
    void sendData(vector<unsigned char> msg) override;
    void writeBytesSafe(const char *bytes, int length) override;
public Q_SLOTS:
    void closeConnection() override;
    void connect2host() override;
    void readyRead() override;
    void connected() override;
    void disconnected() override;
    void connectionTimeout() override;
private:
    void setAddress(QString _multicastAddress, int _multicastPort,
                    QString _host,int _port);
private:
    QUdpSocket *udpSend = nullptr;
    QUdpSocket *udpRecv = nullptr;
    QString host;
    int port;
    QString multicastAddress;
    int multicastPort;

    quint16 m_nNextBlockSize;
};

#endif // TCPMULTICASTLINK_H
