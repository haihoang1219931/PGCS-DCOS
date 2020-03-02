#include "UDPMulticastLink.h"

UDPMulticastLink::UDPMulticastLink(LinkInterface *parent) : LinkInterface(parent){
    printf("Init UDPMulticastLink\r\n");
    status = false;
    udpSend = new QUdpSocket();
    udpRecv = new QUdpSocket();

}
UDPMulticastLink::~UDPMulticastLink(){
    printf("Destroy UDPMulticastLink\r\n");
    if(udpSend != nullptr && udpRecv != nullptr){
        closeConnection();
    }
    udpSend->deleteLater();
    udpRecv->deleteLater();
}
void UDPMulticastLink::connect2host()
{
    printf("UDPMulticastLink::connect2host\r\n");
    udpSend->connectToHost(host, static_cast<unsigned short>(port));
    udpRecv->bind(QHostAddress::AnyIPv4, static_cast<quint16>(multicastPort),
                    QUdpSocket::ShareAddress);
    udpRecv->joinMulticastGroup(QHostAddress(multicastAddress));
    connect(udpRecv, SIGNAL(readyRead()), this, SLOT(readyRead()));
    connect(udpSend, SIGNAL(connected()), this, SLOT(connected()));
    connect(udpSend, SIGNAL(disconnected()), this, SLOT(disconnected()));

}
void UDPMulticastLink::setAddress(QString _multicastAddress, int _multicastPort,
                                  QString _host,int _port){    
    host = _host;
    port = _port;
    multicastAddress = _multicastAddress;
    multicastPort= _multicastPort;    
    printf("host:%s\r\n",host.toStdString().c_str());
    printf("port:%d\r\n",port);
    printf("multicastAddress:%s\r\n",multicastAddress.toStdString().c_str());
    printf("multicastPort:%d\r\n",multicastPort);
}
void UDPMulticastLink::sendData(vector<unsigned char>  msg){
    if(udpSend->isValid()){
        udpSend->write((const char*)msg.data(),(qint64)msg.size());
    }
}
void UDPMulticastLink::writeBytesSafe(const char *bytes, int length){
    if(udpSend->isValid()){
        udpSend->write(bytes,length);
    }
}

void UDPMulticastLink::connectionTimeout()
{
    if(udpSend->state() == QAbstractSocket::ConnectingState)
    {
        udpSend->abort();
        Q_EMIT udpSend->error(QAbstractSocket::SocketTimeoutError);
    }
}

void UDPMulticastLink::connected()
{
    status = true;
    Q_EMIT statusChanged(status);
}
void UDPMulticastLink::disconnected(){
    status = false;
    Q_EMIT statusChanged(status);
}
bool UDPMulticastLink::isOpen(){
    return udpSend->isOpen();
}
void UDPMulticastLink::loadConfig(Config* config){
    name = config->value("Settings:LinkName:Value:data").toString();
    type = config->value("Settings:LinkType:Value:data").toString();
    host = config->value("Settings:PilotIP:Value:data").toString();
    port = config->value("Settings:PilotPortIn:Value:data").toInt();
    multicastAddress = config->value("Settings:MulticastIP:Value:data").toString();
    multicastPort = config->value("Settings:MulticastPort:Value:data").toInt();
    setAddress(multicastAddress,multicastPort,host,port);
}
void UDPMulticastLink::readyRead()
{
    QByteArray buffer;
    if (udpRecv != nullptr) {
        while (udpRecv->hasPendingDatagrams()){
            buffer.resize(int(udpRecv->pendingDatagramSize()));
            udpRecv->readDatagram(buffer.data(), buffer.size());
//            printf("Ready read [%d] bytes\r\n",buffer.size());
//            for(int i=0; i< buffer.size(); i++){
//                printf("0x%02X,",static_cast<unsigned char>(buffer[i]));
//            }
//            printf("\r\n");
            Q_EMIT hasReadSome(buffer);
        }
    }

    return;
}

void UDPMulticastLink::closeConnection()
{
    udpSend->close();
    udpSend->close();
    status = false;
    Q_EMIT statusChanged(status);
    disconnect(udpSend, SIGNAL(connected()), this, SLOT(connected()));
    disconnect(udpSend, SIGNAL(disconnected()), this, SLOT(disconnected()));
    disconnect(udpRecv, SIGNAL(readyRead()), this, SLOT(readyRead()));
}
