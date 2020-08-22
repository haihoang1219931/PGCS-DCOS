#ifndef UDPLINK_H
#define UDPLINK_H

#include <QString>
#include <QUdpSocket>
#include <QDataStream>
#include <QTimer>
#include <vector>
#include <stdio.h>
#include <iostream>
#include "LinkInterface.h"
using namespace std;
class UDPLink : public LinkInterface
{
    Q_OBJECT
public:
    explicit UDPLink(LinkInterface *parent = nullptr);
    virtual ~UDPLink();

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
    void setAddress(QString _host,int _port);
private:
    QUdpSocket *udpSocket;
    QString host;

    int port;
    QHostAddress clientAddress;
    quint16 clientPort;
    quint16 m_nNextBlockSize;
};

#endif // UDPLINK_H
