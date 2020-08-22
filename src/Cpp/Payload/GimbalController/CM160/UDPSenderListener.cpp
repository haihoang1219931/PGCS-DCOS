#include "UDPSenderListener.h"

UdpSenderListener::UdpSenderListener(QObject*parent):
    QObject(parent)
{
    udpClientReceived = new QUdpSocket();
    udpClientSend = new QUdpSocket();
}
UdpSenderListener::UdpSenderListener(QString destination, int sendPort)
{
    remoteEndPoint = IPEndPoint(destination,sendPort);
    udpClientSend = new QUdpSocket();
    udpClientReceived = new QUdpSocket();
//    udpClientReceived->bind(QHostAddress::LocalHost, sendPort);
    connect(udpClientReceived, SIGNAL(readyRead()),
                  this, SLOT(worker_receive()));
}
UdpSenderListener::~UdpSenderListener(){
    delete udpClientReceived;
    delete udpClientSend;
}

void UdpSenderListener::Open(){
    IsConnected = true;
    BeginReceive();
    BeginSend();
}

void UdpSenderListener::Close(){
    IsConnected = false;
    EndReceive();
    EndSend();
    udpClientReceived->close();
    udpClientSend->close();
}

void UdpSenderListener::BeginReceive(){
    //udpClientReceived->moveToThread(&t_receiver);
    //connect(&t_receiver,SIGNAL(started()),this,SLOT(worker_receive()));
    notkillreceive = true;
    //t_receiver.start();
    //t_receiver.setPriority(QThread::LowPriority);
}

void UdpSenderListener::BeginSend(){
    //udpClientSend->moveToThread(&t_sender);
    //connect(&t_sender,SIGNAL(started()),this,SLOT(worker_send()));
    notkillsend = true;
    //t_sender.start();
    //t_sender.setPriority(QThread::LowPriority);
}

void UdpSenderListener::Dispose(){
    notkillreceive = false;
    notkillsend = false;
    EndReceive();
    EndSend();
    udpClientReceived->close();
}

void UdpSenderListener::EndReceive(){
    notkillreceive = false; //have receive worker thread end

    t_receiver.disconnect();
}


void UdpSenderListener::EndSend(){
    notkillsend = false; //have send thread end
    t_sender.disconnect();
}
void UdpSenderListener::worker_receive() {
    try
    {
        QHostAddress ip = QHostAddress::Any; //any ipaddress
        quint16 port = 0;//any port
        while (notkillreceive && udpClientReceived->hasPendingDatagrams())
        {
            try
            {
                unsigned char receivedData[1024];
                qint64 received = udpClientReceived->readDatagram(
                            (char*)receivedData,
                            (qint64)sizeof(receivedData),
                            &ip,
                            &port);
                if(received>0){
                    IPEndPoint source(ip.toString(),port);
                    vector<unsigned char> data;
                    for(int i=0; i< received; i++)
                        data.push_back(receivedData[i]);
                    msgRecBuff.push_back(UdpPayload(data,source));
                    qDebug() << "New Message Broadcast received "<<received<<" bytes from "<<ip<<":"<<port;
                    Q_EMIT onNewMessageBroadcast();
                }

            }
            catch (...)
            {

            }
        }
        qDebug()<<"Done received message";
    }
    catch (...)
    {

    }
}

void UdpSenderListener::worker_send() {
    while (notkillsend)
    {
        QByteArray msg;
        if(msgSendBuff.size() > 0)
            msg = msgSendBuff.at(0);
        try
        {

        }
        catch (...)
        {
            //notkillsend = false;
        }
    }
}

UdpPayload UdpSenderListener::GetMessage()
{
    UdpPayload msg;
    if(msgRecBuff.size()>0){
        msg = msgRecBuff.at(0);
    }
    msgRecBuff.erase(msgRecBuff.begin());
    return msg;
}

void UdpSenderListener::AddMessage(unsigned char *data, qint64 length) const
{
    QByteArray msg;
    for(qint64 i = 0; i< length; i++){
        msg.push_back(data[i]);
    }
    //msgSendBuff.push_back(msg);
}

/// <summary>
/// Send an array of bytes to the remote address' port number
/// </summary>
/// <param name="msg">message to send</param>
/// <returns>The number of bytes to sent, returns -1 if send fails</returns>
qint64 UdpSenderListener::Send(QByteArray msg)
{
    return udpClientReceived->writeDatagram(msg,
                                            QHostAddress("255.255.255.255"),
                                            50000);
}
