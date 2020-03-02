#include "UDPLink.h"

UDPLink::UDPLink(LinkInterface *parent) : LinkInterface(parent){
    printf("Init UDPLink\r\n");
    status = false;
    udpSocket = new QUdpSocket();
}
UDPLink::~UDPLink(){
    printf("Destroy UDPLink\r\n");
    if(udpSocket!= NULL){
        closeConnection();
    }
    udpSocket->deleteLater();
}
void UDPLink::connect2host()
{
    connect(udpSocket, SIGNAL(connected()), this, SLOT(connected()));
    connect(udpSocket, SIGNAL(disconnected()), this, SLOT(disconnected()));
    connect(udpSocket, SIGNAL(readyRead()), this, SLOT(readyRead()));
    udpSocket->bind(QHostAddress::AnyIPv4, static_cast<quint16>(port),
                    QUdpSocket::ShareAddress);
}
void UDPLink::setAddress(QString _host,int _port){
    host = _host;
    port = _port;
    printf("UDPLink connect to %s:%d\r\n",host.toStdString().c_str(),port);
}

void UDPLink::connectionTimeout()
{
    //qDebug() << udpSocket->state();
    if(udpSocket->state() == QAbstractSocket::ConnectingState)
    {
        udpSocket->abort();
        Q_EMIT udpSocket->error(QAbstractSocket::SocketTimeoutError);
    }
}

void UDPLink::connected()
{
    status = true;
    Q_EMIT statusChanged(status);
}
void UDPLink::disconnected(){
    status = false;
    Q_EMIT statusChanged(status);
}
bool UDPLink::isOpen(){
    return udpSocket->isOpen();
}
void UDPLink::loadConfig(Config* config){
    name = config->value("Settings:LinkName:Value:data").toString();
    type = config->value("Settings:LinkType:Value:data").toString();
    host = config->value("Settings:PilotIP:Value:data").toString();
    port = config->value("Settings:PilotPortIn:Value:data").toInt();
    setAddress(host,port);
}
void UDPLink::sendData(vector<unsigned char>  msg){
    if(udpSocket->isValid()){
        udpSocket->writeDatagram((const char*)msg.data(),(qint64)msg.size(),clientAddress,clientPort);
    }
}
void UDPLink::writeBytesSafe(const char *bytes, int length){
    if(udpSocket->isValid()){
//        printf("UDPLink %s to [%s:%d] %d bytes\r\n",__func__,
//               clientAddress.toString().toStdString().c_str(),
//               clientPort,
//               length);
//        for(int i=0; i< length; i++){
//            printf("0x%02X,",static_cast<unsigned char>(bytes[i]));
//        }
//        printf("\r\n");
        udpSocket->writeDatagram(bytes,length,clientAddress,clientPort);
    }
}
void UDPLink::readyRead()
{
    QByteArray buffer;
    if (udpSocket != nullptr) {
        while (udpSocket->hasPendingDatagrams()){
            buffer.resize(int(udpSocket->pendingDatagramSize()));
            udpSocket->readDatagram(buffer.data(), buffer.size(),&clientAddress,&clientPort);
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

void UDPLink::closeConnection()
{
    printf("UDPLink %s udpSocket->state()=[%d]\r\n",__func__,udpSocket->state());
    bool shouldEmit = false;
    switch (udpSocket->state())
    {
        case 0:
            udpSocket->disconnectFromHost();
            shouldEmit = true;
            break;
        case 2:
            udpSocket->abort();
            shouldEmit = true;
            break;
        default:
            udpSocket->abort();
    }

    if (shouldEmit)
    {
        status = false;
        Q_EMIT statusChanged(status);
        disconnect(udpSocket, SIGNAL(connected()), this, SLOT(connected()));
        disconnect(udpSocket, SIGNAL(disconnected()), this, SLOT(disconnected()));
        disconnect(udpSocket, SIGNAL(readyRead()), this, SLOT(readyRead()));
    }    
}
