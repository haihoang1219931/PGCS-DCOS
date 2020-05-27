#include "TelemetryController.h"

TelemetryController::TelemetryController(QObject *parent) : QObject(parent)
{
    m_socket = new QTcpSocket();
    m_timerRequest = new QTimer();
    m_timerRequest->setSingleShot(true);
    m_timerRequest->setInterval(1000);
    connect(m_timerRequest,&QTimer::timeout,this,&TelemetryController::handleRequestTimeout);
    m_timerWriteLog = new QTimer();
    m_timerWriteLog->setSingleShot(false);
    m_timerWriteLog->setInterval(1000);
    connect(m_timerWriteLog,&QTimer::timeout,this,&TelemetryController::handleWriteLogTimeout);
}
TelemetryController::~TelemetryController(){
    if(m_socket->isOpen()){
        m_socket->disconnectFromHost();
        m_socket->deleteLater();
    }
}
void TelemetryController::connectToHost(QString ip, int port,QString user, QString pass){
    printf("Connect to %s:%d\r\n",ip.toStdString().c_str(),port);
    m_socket->connectToHost(ip,port);
    m_ip = ip;
    m_port = port;
    m_user = user;
    m_pass = pass;
    connect(m_socket, SIGNAL(connected()), this, SLOT(onConnected()));
    connect(m_socket, SIGNAL(disconnected()), this, SLOT(onDisconnected()));
    connect(m_socket, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(onTcpError(QAbstractSocket::SocketError)));
    connect(m_socket, SIGNAL(stateChanged(QAbstractSocket::SocketState)), this, SLOT(onTcpStateChanged(QAbstractSocket::SocketState)));
    connect(m_socket,&QTcpSocket::readyRead,this,&TelemetryController::handlePacketReceived);
}
int TelemetryController::getConnectionStatus()
{
    return m_status;
}
void TelemetryController::onConnected()
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

void TelemetryController::onDisconnected()
{
    qDebug() << "Gimbal DIsconnected";
    m_status = 0;
    m_socket->abort();
    m_socket->close();
}

void TelemetryController::onTcpStateChanged(QAbstractSocket::SocketState socketState) {
    //    qDebug() << "State:" << socketState;
}

void TelemetryController::onTcpError(QAbstractSocket::SocketError error) {
    //    qDebug() << "---------------------------------------------";
    //    qDebug() << "Error:" << error;
    m_status = -1;
    QMetaObject::invokeMethod(this, "gimbalRetryConnect", Qt::QueuedConnection);
}

void TelemetryController::gimbalRetryConnect()
{
    //    qDebug() << " Gimbal retry connect";
    m_socket->disconnectFromHost();
    m_socket->connectToHost(m_ip, m_port);
}
void TelemetryController::disconnectFromHost(){
    printf("Disconnect from host\r\n");
    disconnect(m_socket,&QTcpSocket::readyRead,this,&TelemetryController::handlePacketReceived);
    m_socket->disconnectFromHost();
}
void TelemetryController::sendRawData(QString command){
    QByteArray cmd = QByteArray::fromHex(command.toLatin1());
    //    printf("TCP Telemetry Send Data: %s\r\n",command.toStdString().c_str());
    m_socket->write(cmd,cmd.size());
}
void TelemetryController::handlePacketReceived(){
    qint64 byteCount = m_socket->bytesAvailable();
    if (byteCount)
    {
        QByteArray buffer;
        buffer.resize(byteCount);
        m_socket->read(buffer.data(), buffer.size());
        m_buffer.push_back(buffer);
        QString command(m_buffer);
//        printf("TCP Received: %s\r\n",command.toStdString().c_str());
        if(!m_authenticated){
            if(command.contains("UserDevice login: ")){
                if(!m_sendUser){
                    QString response = m_user+ QString("\n");
                    m_socket->write(response.toStdString().c_str(),response.length());
                    m_sendUser = true;
                }
                m_buffer.clear();
            }else if(command.contains("Password: ")){
                if(!m_sendPass){
                    QString response = m_pass+ QString("\n");
                    m_socket->write(response.toStdString().c_str(),response.length());
                    m_sendPass = true;
                }
                m_buffer.clear();
            }else if(command.contains("UserDevice>")){
                if(m_sendPass && m_sendUser){
                    printf("Authenticate successfully\r\n");
                    m_authenticated = true;
                    m_timerRequest->start();
                    m_timerWriteLog->start();
                }
                m_buffer.clear();
            }
        }else{
            parseData(command);
            //            m_buffer.clear();
        }
        if(m_buffer.size()>512){
            m_buffer.remove(0,m_buffer.size()-512);
        }
    }
}
void TelemetryController::handleRequestTimeout(){
    m_timerRequest->stop();
    QString response = "AT+MWSTATUS\n";
    m_socket->write(response.toStdString().c_str(),response.length());
    m_timerRequest->start();
}
void TelemetryController::handleWriteLogTimeout(){
//    printf("[%s:%d] [RSSI:%d] [SNR:%d]\r\n",
//           m_ip.toStdString().c_str(),
//           m_port,
//           m_rssi,
//           m_snr);
    Q_EMIT writeLogTimeout(m_ip,m_snr,m_rssi);
}
void TelemetryController::parseData(QString data){
    if(data.contains("SNR")){
        QRegularExpression re("SNR[ ]+\\(dB\\)[ ]+:[ ]+([-]*\\d+)");
        QRegularExpressionMatch match = re.match(data);
        if (match.hasMatch()) {
            QString value = match.captured(1);
            m_snr = value.toInt();
        }
    }
    if(data.contains("RSSI")){
        QRegularExpression re("RSSI[ ]+\\(dBm\\)[ ]+:[ ]+([-]*\\d+)");
        QRegularExpressionMatch match = re.match(data);
        if (match.hasMatch()) {
            QString value = match.captured(1);
            m_rssi = value.toInt();
        }
    }
}
