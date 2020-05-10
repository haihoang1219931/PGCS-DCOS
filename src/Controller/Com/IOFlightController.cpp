#include "IOFlightController.h"

IOFlightController::IOFlightController(QObject *parent) : QObject(parent)
{
    printf("Init IOFlightController\r\n");
    m_msgManager = new MessageManager();
    m_mutexProcess = new QMutex();
    m_pauseCond = new QWaitCondition();
    m_linkInterfaceManager = new LinkInterfaceManager();

    qRegisterMetaType<mavlink_message_t>("mavlink_message_t");
}
IOFlightController::~IOFlightController(){
    printf("Destroy IOFlightController\r\n");
    if(m_linkInterface!=nullptr){
        if(m_linkInterface->isOpen()){
            m_linkInterface->closeConnection();
        }
        m_linkInterface->deleteLater();
    }
    if(m_msgManager!=nullptr){
        m_msgManager->deleteLater();
    }
    if(m_mutexProcess!=nullptr){
        delete m_mutexProcess;
    }
    if(m_pauseCond!=nullptr){
        delete m_pauseCond;
    }
}
void IOFlightController::handlePacket(QByteArray packet){
#ifdef DEBUG
    printf("received [%d] bytes: ",packet.size());
    for(int i=0; i< packet.size(); i++){
        printf("0X%02X,",static_cast<unsigned char>(packet[i]));
    }
    printf("\r\n");
#endif
//    printf("handlePacket read [%d] bytes\r\n",packet.size());
//    for(int i=0; i< packet.size(); i++){
//        printf("0x%02X,",static_cast<unsigned char>(packet[i]));
//    }
//    printf("\r\n");

#ifdef MAVLINK_PARSE_DEFAULT
    for (int position = 0; position < packet.size(); position++) {
            if (mavlink_parse_char(m_mavlinkChannel, static_cast<uint8_t>(packet[position]), &_message, &_status)) {
                // Got a valid message
#ifdef DEBUG
                printf("Packet Ok at [%d/%d]: SYS: %d, COMP: %d, LEN: %d, MSG ID: %d, CHN ID: %d\n",
                       position,packet.size(),_message.sysid, _message.compid, _message.len, _message.msgid,m_mavlinkChannel);
#endif
                mavlink_status_t* mavlinkStatus = mavlink_get_channel_status(m_mavlinkChannel);
                if (!(mavlinkStatus->flags & MAVLINK_STATUS_FLAG_IN_MAVLINK1) && (mavlinkStatus->flags & MAVLINK_STATUS_FLAG_OUT_MAVLINK1)) {
                    qDebug() << "Switching outbound to mavlink 2.0 due to incoming mavlink 2.0 packet:" << mavlinkStatus << m_mavlinkChannel << mavlinkStatus->flags;
                    mavlinkStatus->flags &= ~MAVLINK_STATUS_FLAG_OUT_MAVLINK1;
                    // Set all links to v2
                    setVersion(200);
                }
                // MAVLink Status
                uint8_t lastSeq = lastIndex[_message.sysid][_message.compid];
                uint8_t expectedSeq = lastSeq + 1;
                // Increase receive counter
                totalReceiveCounter[m_mavlinkChannel]++;
                // Determine what the next expected sequence number is, accounting for
                // never having seen a message for this system/component pair.
                if(firstMessage[_message.sysid][_message.compid]) {
                    firstMessage[_message.sysid][_message.compid] = 0;
                    lastSeq     = _message.seq;
                    expectedSeq = _message.seq;
                }
                // And if we didn't encounter that sequence number, record the error
                //int foo = 0;
                if (_message.seq != expectedSeq)
                {
                    //foo = 1;
                    int lostMessages = 0;
                    //-- Account for overflow during packet loss
                    if(_message.seq < expectedSeq) {
                        lostMessages = (_message.seq + 255) - expectedSeq;
                    } else {
                        lostMessages = _message.seq - expectedSeq;
                    }
                    // Log how many were lost
                    totalLossCounter[m_mavlinkChannel] += static_cast<uint64_t>(lostMessages);
                }

                // And update the last sequence number for this system/component pair
                lastIndex[_message.sysid][_message.compid] = _message.seq;;
                // Calculate new loss ratio
                uint64_t totalSent = totalReceiveCounter[m_mavlinkChannel] + totalLossCounter[m_mavlinkChannel];
                float receiveLossPercent = static_cast<float>(static_cast<double>(totalLossCounter[m_mavlinkChannel]) / static_cast<double>(totalSent));
                receiveLossPercent *= 100.0f;
                #ifdef DEBUG
                printf("lossPercent before = %f\r\n",receiveLossPercent);
#endif
                receiveLossPercent = (receiveLossPercent * 0.5f) + (runningLossPercent[m_mavlinkChannel] * 0.5f);
                runningLossPercent[m_mavlinkChannel] = receiveLossPercent;
                #ifdef DEBUG
                printf("lossPercent after = %f\r\n",receiveLossPercent);
#endif
                if(receiveLossPercent>100) receiveLossPercent = 100;
                else if(receiveLossPercent<0) receiveLossPercent = 0;
                #ifdef DEBUG
                printf("lossPercent after limit = %f\r\n",receiveLossPercent);
#endif
                // Update MAVLink status on every 32th packet
                if ((totalReceiveCounter[m_mavlinkChannel] & 0x1F) == 0) {
                    Q_EMIT mavlinkMessageStatus(_message.sysid, totalSent, totalReceiveCounter[m_mavlinkChannel], totalLossCounter[m_mavlinkChannel], receiveLossPercent);
                }
                // Log file
                uint8_t buf[MAVLINK_MAX_PACKET_LEN+sizeof(quint64)];
                // Write the uint64 time in microseconds in big endian format before the message.
                // This timestamp is saved in UTC time. We are only saving in ms precision because
                // getting more than this isn't possible with Qt without a ton of extra code.
                quint64 time = static_cast<quint64>(QDateTime::currentMSecsSinceEpoch() * 1000);
                qToBigEndian(time, buf);
                // Then write the message to the buffer
                int len = mavlink_msg_to_send_buffer(buf + sizeof(quint64), &_message);

                // Determine how many bytes were written by adding the timestamp size to the message size
                len += sizeof(quint64);

                QByteArray b(reinterpret_cast<const char*>(buf), len);

                LogController::writeBinaryLog(m_logFile,b);
                // Send signal to other component
                Q_EMIT messageReceived(_message);
                memset(&_status,  0, sizeof(_status));
                memset(&_message, 0, sizeof(_message));
            }
    }
#endif
}
void IOFlightController::setVersion(unsigned version)
{
    mavlink_status_t* mavlinkStatus = mavlink_get_channel_status(m_mavlinkChannel);
    // Set flags for version
    if (version < 200) {
        mavlinkStatus->flags |= MAVLINK_STATUS_FLAG_OUT_MAVLINK1;
    } else {
        mavlinkStatus->flags &= ~MAVLINK_STATUS_FLAG_OUT_MAVLINK1;
    }

    _current_version = version;
}
QString IOFlightController::getLogFile(){
    return m_logFile;
}
bool IOFlightController::isConnected(){
    return m_linkInterface->isOpen();
}
void IOFlightController::loadConfig(Config* linkConfig){
    if(linkConfig->value("Settings:LinkType:Value:data").toString() == "TCP"){
        m_linkInterface = m_linkInterfaceManager->linkForAPConnection(
                    LinkInterfaceManager::CONNECTION_TYPE::MAV_TCP);
    }else if(linkConfig->value("Settings:LinkType:Value:data").toString() == "RAGAS"){
        m_linkInterface = m_linkInterfaceManager->linkForAPConnection(
                    LinkInterfaceManager::CONNECTION_TYPE::MAV_RAGAS);
    }else if(linkConfig->value("Settings:LinkType:Value:data").toString() == "SERIAL"){
        m_linkInterface = m_linkInterfaceManager->linkForAPConnection(
                    LinkInterfaceManager::CONNECTION_TYPE::MAV_SERIAL);
    }else if(linkConfig->value("Settings:LinkType:Value:data").toString() == "UDP"){
        m_linkInterface = m_linkInterfaceManager->linkForAPConnection(
                    LinkInterfaceManager::CONNECTION_TYPE::MAV_UDP);
    }
    if(m_linkInterface != nullptr){
        connect(m_linkInterface, SIGNAL(hasReadSome(QByteArray)), this, SLOT(handlePacket(QByteArray)));
        m_linkInterface->loadConfig(linkConfig);
        system("/bin/mkdir -p logs");
        m_logFile = QString::fromStdString("logs/"+
                                           linkConfig->value("Settings:LinkName:Value:data").toString().toStdString()+" "+
                                           FileController::get_day()+" "+FileController::get_time()+".tlog");
    }
}
void IOFlightController::connectLink(){
    m_linkInterface->connect2host();
}
void IOFlightController::disConnectLink(){
    m_linkInterface->closeConnection();
}
void IOFlightController::pause(bool _pause){
    if(_pause == true){
            m_mutexProcess->lock();
            m_pause = true;
            m_mutexProcess->unlock();
        }else{
            m_mutexProcess->lock();
            m_pause = false;
            m_mutexProcess->unlock();
            m_pauseCond->wakeAll();
        }
}
void IOFlightController::msleep(int ms){
#ifdef __linux__
    //linux code goes here
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000 * 1000 };
    nanosleep(&ts, NULL);
#elif _WIN32
    // windows code goes here
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#else

#endif
}
#if (defined __QNX__) | (defined __QNXNTO__)
uint64_t IOFlightController::microsSinceEpoch()
{

    struct timespec time;

    uint64_t micros = 0;

    clock_gettime(CLOCK_REALTIME, &time);
    micros = (uint64_t)time.tv_sec * 1000000 + time.tv_nsec/1000;

    return micros;
}
#else
uint64_t IOFlightController::microsSinceEpoch()
{

    struct timeval tv;

    uint64_t micros = 0;

    gettimeofday(&tv, NULL);
    micros =  ((uint64_t)tv.tv_sec) * 1000000 + tv.tv_usec;

    return micros;
}
#endif
