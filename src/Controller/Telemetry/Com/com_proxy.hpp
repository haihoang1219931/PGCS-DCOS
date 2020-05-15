/**
 * ===========================================================
 * Project:
 * Module: Network Communication
 * Module Short Description:
 * Author: Trung Ng
 * Date: 05/12/2020
 * Viettel Aerospace Institude - Viettel Group
 * ===========================================================
 */


#ifndef SOCKET_RECEIVER_SERVICES_HPP
#define SOCKET_RECEIVER_SERVICES_HPP

/** ///////////////////////////////////////////////////////////////////
 *  //  Include Libs
 *  //
 */

//=============== Including preloading C++ Libraries ===== //
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <string.h>
#include <string>
#include <queue>
#include <algorithm>
#include <cctype>
#include <string_view>

//============== Including Qt lib ====================//
#include <QObject>
#include <QThread>
#include <QDebug>
#include <QTimer>

//============= Including Custom Lib ================//
#include "Network/Socket.hpp"
#include "socket_input_params.hpp"
#include "../LogFile/log_file.hpp"
/**
 * //
 * // Include Done
 * //////////////////////////////////////////////////////////////// */



/** /////////////////////////////////////////////////////////////////
 *  // VideoStreamingReceiver Proxy Declaration
 *  //
 *  //
 */

namespace Proxy
{
    class Com : public QThread
    {
        Q_OBJECT
        public:
            /**
             * @brief createInstance: Create single receiver instance
             * @param sockType
             * @param bindPort
             * @return
             */
            static Com* createInstance(Worker::ComRole comRole, Worker::SocketType sockType, uint16_t bindPort);

            /**
             * @brief createInstance: Create multicast receiver instance
             * @param multicastAddress
             * @param multicastPort
             * @return
             */
            static Com* createInstance(Worker::ComRole comRole, const char* multicastAddress, uint16_t multicastPort);

            /**
             * @brief sendData: Send data to remote server
             * @param key
             * @param data
             * @param serverIp
             * @param targetPort
             * @return
             */
            Q_INVOKABLE virtual bool sendData(QString data, const QString serverIp, int targetPort) = 0;

            /**
             * @brief setTCPServerInfo
             * @param host
             * @param port
             * @param user
             * @param pass
             */
            Q_INVOKABLE virtual void setTCPServerInfo(QString host, int port, QString user, QString pass) = 0;

            /**
             * @brief run: Loop over thread
             */
             virtual void run() = 0;

        public Q_SLOTS:
            /**
             * @brief waitingRequestAuth
             */
            virtual void waitingRequestAuth() = 0;

            /**
             * @brief getDataInfo: send command to get RSSI, SNR, DISTANCE
             */
            virtual void requestDataInfo() = 0;
        Q_SIGNALS:
            /**
             * @brief dataReceived: Received data after successfully authenciated.
             * @param data
             */
            void dataReceived(QString srcAddr, QString dataType, int value);

        protected:
            Com(QObject* parent) : QThread(parent){}
            virtual ~Com() {}
            //--- Prevent copy constructor and copy assignment operator
            Com( const Com& ) = delete;
            Com& operator =( const Com& ) = delete;

        private:
            static Com* inst;
    };
}

/**
 * //
 * // Declaration Done
 * //////////////////////////////////////////////////////////////// */

#endif // SOCKET_RECEIVER_SERVICES_HPP
