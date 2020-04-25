#include "SensorController.h"

SensorController::SensorController(QObject *parent) : QObject(parent)
{
    m_socket = new QTcpSocket();
}
SensorController::~SensorController(){
    if(m_socket->isOpen()){
        m_socket->disconnectFromHost();
        m_socket->deleteLater();
    }
}
void SensorController::connectToHost(QString ip, int port){
    printf("Connect to %s:%d\r\n",ip.toStdString().c_str(),port);
    m_socket->connectToHost(ip,port);
    m_ip = ip;
    m_port = port;
    connect(m_socket, SIGNAL(connected()), this, SLOT(onConnected()));
    connect(m_socket, SIGNAL(disconnected()), this, SLOT(onDisconnected()));
    connect(m_socket, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(onTcpError(QAbstractSocket::SocketError)));
    connect(m_socket, SIGNAL(stateChanged(QAbstractSocket::SocketState)), this, SLOT(onTcpStateChanged(QAbstractSocket::SocketState)));
    connect(m_socket,&QTcpSocket::readyRead,this,&SensorController::handlePacketReceived);
}
int SensorController::getConnectionStatus()
{
    return m_status;
}
void SensorController::onConnected()
{
    qDebug() << "Gimbal Connected";
    m_status = 1;
    int enableKeepAlive = 1;
    int fd = m_socket->socketDescriptor();
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &enableKeepAlive, sizeof(enableKeepAlive));

    int maxIdle = 5; /* seconds */
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &maxIdle, sizeof(maxIdle));

    int count = 2; // send up to 3 keepalive packets out, then disconnect if no response
    setsockopt(fd, SOL_TCP, TCP_KEEPCNT, &count, sizeof(count));

    int interval = 1; // send a keepalive packet out every 2 seconds (after the 5 second idle period)
    setsockopt(fd, SOL_TCP, TCP_KEEPINTVL, &interval, sizeof(interval));
}

void SensorController::onDisconnected()
{
    qDebug() << "Gimbal DIsconnected";
    m_status = 0;
    m_socket->abort();
    m_socket->close();
}

void SensorController::onTcpStateChanged(QAbstractSocket::SocketState socketState) {
//    qDebug() << "State:" << socketState;
}

void SensorController::onTcpError(QAbstractSocket::SocketError error) {
//    qDebug() << "---------------------------------------------";
//    qDebug() << "Error:" << error;
    m_status = -1;
    QMetaObject::invokeMethod(this, "gimbalRetryConnect", Qt::QueuedConnection);
}

void SensorController::gimbalRetryConnect()
{
//    qDebug() << " Gimbal retry connect";
    m_socket->disconnectFromHost();
    m_socket->connectToHost(m_ip, m_port);
}
void SensorController::disconnectFromHost(){
    printf("Disconnect from host\r\n");
    disconnect(m_socket,&QTcpSocket::readyRead,this,&SensorController::handlePacketReceived);
    m_socket->disconnectFromHost();
}
void SensorController::sendRawData(QString command){
    QByteArray cmd = QByteArray::fromHex(command.toLatin1());
//    printf("TCP Sensor Send Data: ");
//        for(int i = 0; i < cmd.size(); i++){
//            printf("0x%02X, ", (unsigned char)cmd[i]);
//    }
//    printf("\r\n");
    Q_EMIT dataSend(command);
    m_socket->write(cmd,cmd.size());
}
void SensorController::handlePacketReceived(){
    unsigned char receivedDataArray[1024];
//    while(m_socket->)
    qint64 byteCount = m_socket->bytesAvailable();
    if (byteCount)
    {
//            printf("TCP recevied %d byte\r\n",byteCount);
        QByteArray buffer;
        buffer.resize(byteCount);
        m_socket->read(buffer.data(), buffer.size());
//        printf("TCP Sensor Received Data: ");
//            for(int i = 0; i < buffer.size(); i++){
//                printf("0x%02X, ", (unsigned char)buffer.at(i));
//        }
//        printf("\r\n");
        Q_EMIT dataReceived(QString(buffer.toHex()).toUpper());
    }

}
