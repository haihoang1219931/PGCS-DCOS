#ifndef CLIENTSTUFF_H
#define CLIENTSTUFF_H

#include <QString>
#include <QTcpSocket>
#include <QDataStream>
#include <QTimer>
#include <vector>
#include <stdio.h>
#include <iostream>
using namespace std;
class ClientStuff : public QObject
{
        Q_OBJECT

    public:
        explicit ClientStuff(QObject *parent = nullptr);
        explicit ClientStuff(const QString host, int port);
        virtual ~ClientStuff();
        QTcpSocket *tcpSocket;
        bool getStatus();

        void setAddress(QString _host, int _port);
        void sendData(vector<unsigned char> msg);
        void Send(const char* msg,int len);

    public Q_SLOTS:
        void closeConnection();
        void connect2host();

    Q_SIGNALS:
        void statusChanged(bool);
        void hasReadSome(QByteArray msg);

    private Q_SLOTS:
        void readyRead();
        void connected();
        void connectionTimeout();

    private:
        QString host;
        int port;
        bool status;
        quint16 m_nNextBlockSize;
        QTimer *timeoutTimer;
};

#endif // CLIENTSTUFF_H
