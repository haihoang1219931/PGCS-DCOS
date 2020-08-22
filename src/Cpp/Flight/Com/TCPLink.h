#ifndef TCPLINK_H
#define TCPLINK_H

#include <QString>
#include <QTcpSocket>
#include <QDataStream>
#include <QTimer>
#include <vector>
#include <stdio.h>
#include <iostream>
#include "LinkInterface.h"
using namespace std;
class TCPLink : public LinkInterface
{
    Q_OBJECT
public:
    explicit TCPLink(LinkInterface *parent = nullptr);
    virtual ~TCPLink();

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
    QTcpSocket *tcpSocket;
    QString host;
    int port;

    quint16 m_nNextBlockSize;
};

#endif // TCPLINK_H
