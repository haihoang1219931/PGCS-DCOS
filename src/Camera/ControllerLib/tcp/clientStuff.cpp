#include "clientStuff.h"

ClientStuff::ClientStuff(QObject *parent) : QObject(parent){
    status = false;
    tcpSocket = new QTcpSocket();

    timeoutTimer = new QTimer();
    timeoutTimer->setSingleShot(true);
    connect(timeoutTimer, &QTimer::timeout, this, &ClientStuff::connectionTimeout);

    connect(tcpSocket, &QTcpSocket::disconnected, this, &ClientStuff::closeConnection);
}

ClientStuff::ClientStuff(const QString hostAddress, int portNumber) : QObject(), m_nNextBlockSize(0)
{
    status = false;
    tcpSocket = new QTcpSocket();

    host = hostAddress;
    port = portNumber;

    timeoutTimer = new QTimer();
    timeoutTimer->setSingleShot(true);
    connect(timeoutTimer, &QTimer::timeout, this, &ClientStuff::connectionTimeout);

    connect(tcpSocket, &QTcpSocket::disconnected, this, &ClientStuff::closeConnection);
}
ClientStuff::~ClientStuff(){
    printf("Destroy ClientStuff\r\n");
    if(tcpSocket!= NULL){
        closeConnection();
    }
    tcpSocket->deleteLater();
}
void ClientStuff::connect2host()
{
    timeoutTimer->start(3000);

    tcpSocket->connectToHost(host, port);
    connect(tcpSocket, &QTcpSocket::connected, this, &ClientStuff::connected);
    connect(tcpSocket, &QTcpSocket::readyRead, this, &ClientStuff::readyRead);
}
void ClientStuff::setAddress(QString _host,int _port){
    host = _host;
    port = _port;
}
void ClientStuff::sendData(vector<unsigned char>  msg){
//    printf("Send: ");
//    for(int i=0; i< msg.size(); i++){
//        printf(" %02x",(unsigned char)msg[i]);
//    }
//    printf("\r\n");
    if(tcpSocket->isValid()){

        tcpSocket->write((const char*)msg.data(),(qint64)msg.size());
    }
}
void ClientStuff::Send(const char* msg,int len){
//    printf("Send: ");
//    for(int i=0; i< msg.size(); i++){
//        printf(" %02x",(unsigned char)msg[i]);
//    }
//    printf("\r\n");
    if(tcpSocket->isValid()){

        tcpSocket->write((const char*)msg,len);
    }
}

void ClientStuff::connectionTimeout()
{
    //qDebug() << tcpSocket->state();
    if(tcpSocket->state() == QAbstractSocket::ConnectingState)
    {
        tcpSocket->abort();
        Q_EMIT tcpSocket->error(QAbstractSocket::SocketTimeoutError);
    }
}

void ClientStuff::connected()
{
    printf("Gimbal Connected");
    status = true;
    Q_EMIT statusChanged(status);
}

bool ClientStuff::getStatus() {return status;}

void ClientStuff::readyRead()
{
    unsigned char receivedData[1024];
    qint64 received = tcpSocket->read((char *)receivedData,sizeof(receivedData));
    QByteArray str((const char*)receivedData,received);
    Q_EMIT hasReadSome(str);
//    printf("Gimbal return %d bytes\r\n",str.size());
    return;
}

//void ClientStuff::gotDisconnection()
//{
//    status = false;
//    Q_EMIT statusChanged(status);
//}

void ClientStuff::closeConnection()
{
    timeoutTimer->stop();

    //qDebug() << tcpSocket->state();
    disconnect(tcpSocket, &QTcpSocket::connected, 0, 0);
    disconnect(tcpSocket, &QTcpSocket::readyRead, 0, 0);

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
    }
}
