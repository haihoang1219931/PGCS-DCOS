#ifndef IOFLIGHTCONTROLLER_H
#define IOFLIGHTCONTROLLER_H

#include <QObject>
#include <QTcpSocket>
#include <QMutex>
#include <QWaitCondition>
#include <QTimer>
#include <deque>
#include <thread>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <QDateTime>
#include <QtEndian>
#include <time.h>
#if (defined __QNX__) | (defined __QNXNTO__)
/* QNX specific headers */
#include <unix.h>
#else
/* Linux / MacOS POSIX timer headers */
#include <sys/time.h>
#include <time.h>
#include <arpa/inet.h>
#include <stdbool.h> /* required for the definition of bool in C99 */
#endif
#include "LinkInterfaceManager.h"
#include "LinkInterface.h"
#include "MessageManager.h"
#include "../../Log/LogController.h"
#include "../../Files/FileControler.h"
//#define DEBUG
#define BUFFER_LENGTH 8192
//#define MAVLINK_PARSE_CUSTOM
#define MAVLINK_PARSE_DEFAULT
Q_DECLARE_METATYPE(mavlink_message_t)
class IOFlightController  : public QObject
{
    Q_OBJECT
public:
    explicit IOFlightController(QObject *parent = nullptr);
    virtual ~IOFlightController();
    uint8_t mavlinkChannel(){return m_mavlinkChannel;}
    uint8_t systemId(){return m_systemId;}
    uint8_t componentId(){return m_componentId;}
    /// Set protocol version
    void setVersion(unsigned version);
    QString getLogFile();
Q_SIGNALS:
    void receivePacket(QByteArray msg);
    void messageReceived(mavlink_message_t message);
    void mavlinkMessageStatus(int uasId, uint64_t totalSent, uint64_t totalReceived, uint64_t totalLoss, float lossPercent);
public Q_SLOTS:
    void loadConfig(Config* linkConfig);
    void connectLink();
    void disConnectLink();
    void pause(bool _pause);
    void handlePacket(QByteArray packet);
    bool isConnected();
    LinkInterface * getInterface(){ return m_linkInterface;}
public:
    QMutex* m_mutexProcess = nullptr;
    QWaitCondition *m_pauseCond = nullptr;
    LinkInterface *m_linkInterface = nullptr;
    LinkInterfaceManager *m_linkInterfaceManager = nullptr;
    MessageManager *m_msgManager = nullptr;
    QString m_logFile;
    bool m_pause = false;
    bool m_stop = false;
    int m_timeCount = 0;
    const int SLEEP_TIME = 10;
    // mavlink
    bool        m_enable_version_check;                         ///< Enable checking of version match of MAV and QGC
    uint8_t     lastIndex[256][256];                            ///< Store the last received sequence ID for each system/componenet pair
    uint8_t     firstMessage[256][256];                         ///< First message flag
    uint64_t    totalReceiveCounter[MAVLINK_COMM_NUM_BUFFERS];  ///< The total number of successfully received messages
    uint64_t    totalLossCounter[MAVLINK_COMM_NUM_BUFFERS];     ///< Total messages lost during transmission.
    float       runningLossPercent[MAVLINK_COMM_NUM_BUFFERS];   ///< Loss rate
    QByteArray m_queuePacket;
    mavlink_message_t _message;
    mavlink_status_t _status;

    bool        versionMismatchIgnore;
    uint8_t         m_mavlinkChannel = MAVLINK_COMM_2;
    uint8_t         m_systemId = 255;
    uint8_t         m_componentId = 0;
    unsigned    _current_version;
    int         _radio_version_mismatch_count;
public:
    uint64_t microsSinceEpoch();
    void msleep(int ms);
};

#endif // IOFLIGHTCONTROLLER_H
