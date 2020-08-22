#ifndef UDPSENDERLISTENER_H
#define UDPSENDERLISTENER_H

#include <QHostAddress>
#include <QUdpSocket>
#include <QThread>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <thread>
#include <mutex>
#include "UDPPayload.h"
using namespace std;
class UdpSenderListener: public QObject
{
    Q_OBJECT
public:
    UdpSenderListener(QObject* parent = nullptr);
    UdpSenderListener(QString destination, int sendPort);
    virtual ~UdpSenderListener();
    QUdpSocket *udpClientSend;
    QUdpSocket *udpClientReceived;
    IPEndPoint remoteEndPoint;
    vector<UdpPayload> msgRecBuff;
    vector<QByteArray> msgSendBuff;
    QThread t_receiver;
    QThread t_sender;
    bool notkillreceive;
    bool notkillsend;
    bool IsConnected;
    void Open();
    void Close();
    void BeginReceive();
    void BeginSend();
    void Dispose();
    void AddMessage(unsigned char *data, qint64 length) const;
    qint64 Send(QByteArray msg);
    UdpPayload GetMessage();
Q_SIGNALS:
    void onNewMessageBroadcast();
public Q_SLOTS:
    void worker_receive();
    void worker_send();
private:
    void EndReceive();
    void EndSend();

};

#endif // UDPSENDERLISTENER_H
