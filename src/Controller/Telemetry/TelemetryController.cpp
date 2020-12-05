#include "TelemetryController.h"

TelemetryController::TelemetryController(QObject *parent) : QObject(parent)
{
    m_socket = new QTcpSocket();
    m_timerRequest = new QTimer();
    m_timerRequest->setSingleShot(true);
    m_timerRequest->setInterval(2000);
    connect(m_timerRequest,&QTimer::timeout,this,&TelemetryController::handleRequestTimeout);

    m_timerWriteLog = new QTimer();
    m_timerWriteLog->setSingleShot(false);
    m_timerWriteLog->setInterval(5000);
    connect(m_timerWriteLog,&QTimer::timeout,this,&TelemetryController::handleWriteLogTimeout);

    m_timerCheck = new QTimer();
    m_timerCheck->setSingleShot(false);
    m_timerCheck->setInterval(5000);
    connect(m_timerCheck,&QTimer::timeout,this,&TelemetryController::handleCheckConnect);
}
TelemetryController::~TelemetryController(){
    if(m_socket->isOpen()){
        m_socket->disconnectFromHost();
        m_socket->deleteLater();
    }
}
void TelemetryController::handleCheckConnect(){
#ifdef DEBUG_FUNC
    printf("m_countLast[%d] m_countParsed[%d]\r\n",
           m_countLast,m_countParsed);
#endif
    int countCurrent = m_countLast;
    if(m_countParsed - countCurrent < 1){
        #ifdef DEBUG_FUNC
        printf("Link lost\r\n");
#endif
        m_linkGood = false;
        m_snr = -1;
        m_rssi = -1;
    }else{
        #ifdef DEBUG_FUNC
        printf("Link good\r\n");
#endif
        m_linkGood = true;
    }
    m_countLast = m_countParsed;
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
    m_timerWriteLog->start();
    m_timerCheck->start();
}
int TelemetryController::getConnectionStatus()
{
    return m_status;
}
void TelemetryController::onConnected()
{
    qDebug() << "Telemetry Connected";
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
    qDebug() << "Telemetry disconnected";
    m_status = 0;
    m_socket->abort();
    m_socket->close();
}

void TelemetryController::onTcpStateChanged(QAbstractSocket::SocketState socketState) {
    qDebug() << "State:" << socketState;
}

void TelemetryController::onTcpError(QAbstractSocket::SocketError error) {
    //    qDebug() << "---------------------------------------------";
    qDebug() << "Error:" << error;
    m_status = -1;
    QMetaObject::invokeMethod(this, "gimbalRetryConnect", Qt::QueuedConnection);
}

void TelemetryController::gimbalRetryConnect()
{
    qDebug() << " Gimbal retry connect";
    disconnectFromHost();
    connectToHost(m_ip, m_port,m_user,m_pass);
}
void TelemetryController::disconnectFromHost(){
    printf("Disconnect from host\r\n");
    disconnect(m_socket, SIGNAL(connected()), this, SLOT(onConnected()));
    disconnect(m_socket, SIGNAL(disconnected()), this, SLOT(onDisconnected()));
    disconnect(m_socket, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(onTcpError(QAbstractSocket::SocketError)));
    disconnect(m_socket, SIGNAL(stateChanged(QAbstractSocket::SocketState)), this, SLOT(onTcpStateChanged(QAbstractSocket::SocketState)));
    disconnect(m_socket,&QTcpSocket::readyRead,this,&TelemetryController::handlePacketReceived);
    m_timerRequest->stop();
    m_timerWriteLog->stop();
    m_timerCheck->stop();
    m_socket->disconnectFromHost();
    m_buffer.clear();
    m_authenticated = false;
    m_sendUser = false;
    m_sendPass = false;
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
                }
                m_buffer.clear();
            }
        }else{
            if(command.contains("UserDevice>")){
                m_timerRequest->start();
                parseData(command);
                m_buffer.clear();
            }
        }
    }
}
void TelemetryController::handleRequestTimeout(){
    QString response = "AT+MWSTATUS\n";
    m_socket->write(response.toStdString().c_str(),response.length());
    #ifdef DEBUG_FUNC
    printf("Request Data ===========================\r\n");
#endif
}
void TelemetryController::handleWriteLogTimeout(){
    #ifdef DEBUG_FUNC
    printf("[%s:%d] [%s] [RSSI:%d] [SNR:%d]\r\n",
               m_ip.toStdString().c_str(),
               m_port,
               m_linkGood?"LINK GOOD":"LINK LOST",
               m_rssi,
               m_snr);
#endif
    Q_EMIT writeLogTimeout(m_ip,m_snr,m_rssi);
}
void TelemetryController::parseData(QString data){
    m_countParsed++;
    if(m_countParsed > 5){
        m_countLast = m_countLast-m_countParsed;
        m_countParsed = 0;
    }
    #ifdef DEBUG_FUNC
//    printf("Parse data\r\n");
#endif
    //    printf("TCP Received =========================== : %s\r\n",data.toStdString().c_str());
    if(data.contains("SNR")){
        QRegularExpression re("SNR[ ]+\\(dB\\)[ ]+:[ ]+([-]*\\d+)");
        QRegularExpressionMatch match = re.match(data);
        if (match.hasMatch()) {
            QString value = match.captured(1);
            m_snr = value.toInt();
        }
    }else{
        m_snr = -1;
    }
    if(data.contains("RSSI")){
        QRegularExpression re("RSSI[ ]+\\(dBm\\)[ ]+:[ ]+([-]*\\d+)");
        QRegularExpressionMatch match = re.match(data);
        if (match.hasMatch()) {
            QString value = match.captured(1);
            m_rssi = value.toInt();
        }
    }else{
        m_rssi = -1;
    }
}
