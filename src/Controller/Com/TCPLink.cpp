#include "TCPLink.h"

TCPLink::TCPLink(LinkInterface *parent) : LinkInterface(parent){
    printf("Init TCPLink\r\n");
    status = false;
    tcpSocket = new QTcpSocket();

}
TCPLink::~TCPLink(){
    printf("Destroy TCPLink\r\n");
    if(tcpSocket!= NULL){
        closeConnection();
    }
    tcpSocket->deleteLater();
}
void TCPLink::connect2host()
{
    connect(tcpSocket, SIGNAL(connected()), this, SLOT(connected()));
    connect(tcpSocket, SIGNAL(disconnected()), this, SLOT(disconnected()));
    connect(tcpSocket, SIGNAL(readyRead()), this, SLOT(readyRead()));
    tcpSocket->connectToHost(host, static_cast<quint16>(port));
}
void TCPLink::setAddress(QString _host,int _port){
    host = _host;
    port = _port;
    printf("TCPLink connect to %s:%d\r\n",host.toStdString().c_str(),port);
}

void TCPLink::connectionTimeout()
{
    //qDebug() << tcpSocket->state();
    if(tcpSocket->state() == QAbstractSocket::ConnectingState)
    {
        tcpSocket->abort();
        Q_EMIT tcpSocket->error(QAbstractSocket::SocketTimeoutError);
    }
}

void TCPLink::connected()
{
    status = true;
    Q_EMIT statusChanged(status);
}
void TCPLink::disconnected(){
    status = false;
    Q_EMIT statusChanged(status);
}
bool TCPLink::isOpen(){
    return tcpSocket->isOpen();
}
void TCPLink::loadConfig(Config* config){
    name = config->value("Settings:LinkName:Value:data").toString();
    type = config->value("Settings:LinkType:Value:data").toString();
    host = config->value("Settings:PilotIP:Value:data").toString();
    port = config->value("Settings:PilotPortIn:Value:data").toInt();
    setAddress(host,port);
}
void TCPLink::sendData(vector<unsigned char>  msg){
    if(tcpSocket->isValid()){
        tcpSocket->write((const char*)msg.data(),(qint64)msg.size());
    }
}
void TCPLink::writeBytesSafe(const char *bytes, int length){
    if(tcpSocket->isValid()){
        tcpSocket->write(bytes,length);
    }
}
void TCPLink::readyRead()
{
    if (tcpSocket) {
        qint64 byteCount = tcpSocket->bytesAvailable();
        if (byteCount)
        {
//            printf("TCP recevied %d byte\r\n",byteCount);
            QByteArray buffer;
            buffer.resize(byteCount);
            tcpSocket->read(buffer.data(), buffer.size());
            Q_EMIT hasReadSome(buffer);
        }
    }
    return;
}

void TCPLink::closeConnection()
{
    printf("TCPLINK %s tcpSocket->state()=[%d]\r\n",__func__,tcpSocket->state());
    bool shouldEmit = false;
    switch (tcpSocket->state())
    {
        case 0:
            tcpSocket->disconnectFromHost();
            shouldEmit = true;
            break;
        case 2:
            tcpSocket->abort();
            shouldEmit = true;
            break;
        default:
            tcpSocket->abort();
    }

    if (shouldEmit)
    {
        status = false;
        Q_EMIT statusChanged(status);
        disconnect(tcpSocket, SIGNAL(connected()), this, SLOT(connected()));
        disconnect(tcpSocket, SIGNAL(disconnected()), this, SLOT(disconnected()));
        disconnect(tcpSocket, SIGNAL(readyRead()), this, SLOT(readyRead()));
    }    
}
