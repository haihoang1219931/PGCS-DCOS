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

#include "com_proxy.hpp"

#define LOG_COM_PROXY "[com_proxy.cpp] - "

namespace Proxy
{
    Com* Com::inst = nullptr;
    class ComImpl : public Com
    {
        public:
            /**
             * @brief ComImpl: Constructor and destructor
             * @param enableMulticast
             */
            ComImpl(const Worker::ComRole comRole, const Worker::SocketType sockType, uint16_t bindPort, QObject* parent = nullptr);

            /**
             * @brief SocketReceiverProxyImpl
             * @param multicastAddress
             * @param multicastPort
             * @param parent
             */
            ComImpl(const Worker::ComRole comRole, const char* multicastAddress, uint16_t multicastPort, QObject* parent = nullptr);

            /**
             * @brief ~ComImpl
             */
            ~ComImpl();

            /**
             * @brief sendData: Implementation of sendData. Send data requested to remote server
             * @param key
             * @param data
             * @param serverIp
             * @param targetPort
             * @return
             */
            bool sendData(QString data, const QString serverIp, int targetPort) override;

            /**
             * @brief setTCPServerInfo
             * @param host
             * @param port
             */
            void setTCPServerInfo(QString host, int port, QString user, QString pass) override;

            /**
             * @brief run: Run thread to receive data from this socket.
             */
            void run();

            /**
             * @brief waitingRequestAuth
             */
            void waitingRequestAuth() override;

            /**
             * @brief getDataInfo: send command to get RSSI, SNR, DISTANCE
             */
            void requestDataInfo() override;

        protected:
            /**
             * @brief setUp
             * @param address
             * @param port
             * @param multicast
             * @return
             */
            bool setUp(const char* address, uint16_t port, bool multicast=false);

            /**
             * @brief send: Do send packet to server
             * @param key
             * @param buffer
             */
            bool send(const char* buffer, uint32_t serverIp, uint16_t targetPort);

            /**
             * @brief reqParse
             */
            void reqParse();

            /**
             * @brief attactLogFile
             * @param logFile
             */
            void attachLogFile(Utils::LogFile* logFile);

            /**
             * @brief isInt
             * @param str
             * @return
             */
            bool isInt(std::string str) {
                str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
                return std::all_of(str.cbegin(), str.cend(), ::isdigit);
            }

        private:
            Worker::Socket* mSock;
            Worker::SocketType mSockType;
            Worker::ComRole mComRole;
            std::vector<unsigned char> mDataQueue;
            Utils::LogFile* mLogFile = nullptr;
            QTimer* timer;
            QTimer* timerReqInfos;
            QString mTargetHost = "";
            QString mTargetUser = "admin";
            QString mTargetPass = "ttuav";
            int mTargetPort = 0;
            bool mAuthenticated = false;
            bool mTcpConnected = false;
    };

    //---------------------- Definition
    ComImpl::ComImpl(const Worker::ComRole comRole, const Worker::SocketType sockType, uint16_t bindPort, QObject* parent)
        : Com(parent), mSockType(sockType), mComRole(comRole)
    {
        mSock = Worker::Socket::Create(sockType);
        mSock->setEnableMulticast(false);
        if (mComRole == Worker::ComRole::SERVER && !setUp("", bindPort)) {
            qDebug(LOG_COM_PROXY "Failed to bind and connect to socket. !");
            return;
        }

        //--- TCP receiver
//        mainTcpReciverThread = new QTimer();
//        mainTcpReciverThread->setInterval(1000);
//        mainTcpReciverThread->setSingleShot(false);
//        connect(mainTcpReciverThread, &QTimer::timeout, this, &Proxy::ComImpl::run);
//        mainTcpReciverThread->start();

        //--- Send data to wait for server authentication request
        timer = new QTimer();
        timer->setInterval(1000);
        timer->setSingleShot(false);
        connect(timer, &QTimer::timeout, this, &Proxy::ComImpl::waitingRequestAuth);
        timer->start();

        //--- Send request to get RSSI, SNR, DISTANCE in sequence
        timerReqInfos = new QTimer();
        timerReqInfos->setInterval(2000);
        timerReqInfos->setSingleShot(false);
        connect(timerReqInfos, &QTimer::timeout, this, &Proxy::ComImpl::requestDataInfo);
        timerReqInfos->start();
    }

    ComImpl::ComImpl(const Worker::ComRole comRole, const char* multicastAddress, uint16_t multicastPort, QObject* parent)
        : Com(parent), mComRole(comRole)
    {
        mSock = Worker::Socket::Create(Worker::SocketType::SOCKET_UDP);
        mSock->setEnableMulticast(true);
        if (mComRole == Worker::ComRole::SERVER && !setUp(multicastAddress, multicastPort,true)) {
            qDebug(LOG_COM_PROXY "Failed to bind and connect to socket. !");
            return;
        }
    }

    ComImpl::~ComImpl()
    {
        delete mSock;
        delete mLogFile;
    }

    void ComImpl::setTCPServerInfo(QString host, int port, QString user, QString pass) {
        mTargetHost = host;
        mTargetPort = port;
        if (!mLogFile) {
            this->attachLogFile(new Utils::LogFile(mTargetHost.toStdString() + ".txt"));
        }

        if( mSockType == Worker::SocketType::SOCKET_TCP && mSock->Connect(mTargetHost.toStdString().c_str(), (uint16_t) mTargetPort))
        {
            //qDebug(LOG_COM_PROXY "Socket haven't connected to %s - %d yet. \t\n", serverIp.toStdString().c_str(), targetPort);
            //return false;
            mTcpConnected = true;
        }
    }

    bool ComImpl::setUp(const char *address, uint16_t port,bool multicast)
    {
        if( mSock != NULL )
        {
            if( mSock->Bind(port) )
            {
                if( multicast )
                {
                    mSock->joinMulticast(address);
                }
                //--- If UDP it will return true immidiately without doing anything (do nothing)
                return mSock->Accept(5);
            }
        }
        return false;
        return true;
    }

    void ComImpl::attachLogFile(Utils::LogFile *logFile) {
        mLogFile = logFile;
    }

    bool ComImpl::sendData(QString data, const QString serverIp, int targetPort)
    {
        uint32_t ipAddress = 0;

        if (!Worker::IPv4Address(serverIp.toStdString().c_str(), &ipAddress))
            return false;

        //if( mSockType == Worker::SocketType::SOCKET_TCP && !mSock->Connect(ipAddress, (uint16_t) targetPort))
        //{
        //    qDebug(LOG_COM_PROXY "Socket haven't connected to %s - %d yet. \t\n", serverIp.toStdString().c_str(), targetPort);
        //    return false;
        //}
        return send(data.toStdString().c_str(), ipAddress, (uint16_t) targetPort);
    }

    bool ComImpl::send(const char *buffer, uint32_t serverIp, uint16_t targetPort)
    {
         //qDebug(LOG_COM_PROXY "Packet sent: [%s-%d] to %s:%d",
         //      buffer, (int) strlen(buffer), Worker::IPv4AddressStr(serverIp).c_str(), targetPort);
        if( !mSock->Send( static_cast<const void*>(buffer), strlen(buffer), serverIp, targetPort ) )
        {
            if (mTcpConnected) {
                mSock->Connect(serverIp, (uint16_t) targetPort);
            }
            qDebug(LOG_COM_PROXY "Failed to send data to remote address. \t\n" );
            return false;
        }
        return true;
    }

    void ComImpl::run()
    {
        unsigned char dataRe[1400];
        uint32_t iSrcIpAddress;
        uint16_t iSrcPort;
        //bool authentiated = false;
        while (true) {
            size_t byteReceived = mSock->Recieve(dataRe, 1400, &iSrcIpAddress, &iSrcPort);
            if( byteReceived > 0)
            {
                std::string dataReStr(dataRe, dataRe + byteReceived);
                //qDebug(LOG_COM_PROXY "Data received: %s", dataReStr.c_str());
                if (mAuthenticated) {
                    //Q_EMIT dataReceived(QString::fromStdString(dataReStr), 1);
                    for (auto i : dataReStr) {
                        mDataQueue.push_back(i);
                    }
                    this->reqParse();
                }

                if ((int) dataReStr.find("login") != -1) {
                    //Q_EMIT authRequestUsername();
                    this->sendData(mTargetUser+QString("\n"), mTargetHost, (int) mTargetPort);
                }

                if ((int) dataReStr.find("Password") != -1) {
                    //Q_EMIT authRequestPassword();
                    this->sendData(mTargetPass+QString("\n"), mTargetHost, (int) mTargetPort);
                }

                if ((int) dataReStr.find("UserDevice>") != -1) {
                    mAuthenticated = true;

                }
            }
        }
    }

    void ComImpl::reqParse() {
        std::string data(mDataQueue.begin(), mDataQueue.end());
//        qDebug("--------------->1. %s", data.c_str());
        while (mDataQueue.size() > 0 && mDataQueue[0] != 'A') {
            mDataQueue.erase(mDataQueue.begin(), mDataQueue.begin() + 1);
        }

        while(mDataQueue.size() > 1 && mDataQueue[1] != 'T') {
            mDataQueue.erase(mDataQueue.begin(), mDataQueue.begin() + 2);
        }
        //data = std::string(mDataQueue.begin(), mDataQueue.end());
        //qDebug("--------------->2. %s", data.c_str());
        if (mDataQueue.size() >= 31) {
            std::string data(mDataQueue.begin(), mDataQueue.begin() + 31);
//            qDebug("--------------->3. %s", data.c_str());
            if ((int) data.find("AT+MWRSSI") == 0) {
                std::string rSSI = std::string(mDataQueue.begin() + 12, mDataQueue.begin() + 14);
                if (rSSI != "N/A" && isInt(rSSI)) {
                    Q_EMIT dataReceived(mTargetHost, "RSSI", -std::stoi(rSSI));
                    mLogFile->write(rSSI);
                } else {
                    // qDebug("RSSI: N/A");
                    Q_EMIT dataReceived(mTargetHost, "RSSI", -1);
                    mLogFile->write("N/A");
                }
                mDataQueue.erase(mDataQueue.begin(), mDataQueue.begin() + 14);
            }

            if ((int)data.find("AT+MWSNR") == 0) {
                std::string sNR = std::string(mDataQueue.begin() + 10, mDataQueue.begin() + 12);
                if (sNR != "N/" && isInt(sNR)) {
                    Q_EMIT dataReceived(mTargetHost, "SNR", std::stoi(sNR));
                    mLogFile->write(", " + sNR);
                    mDataQueue.erase(mDataQueue.begin(), mDataQueue.begin() + 12);

                } else {
                    // qDebug("SNR: N/A");
                    Q_EMIT dataReceived(mTargetHost, "SNR", -1);
                    mLogFile->write(", N/A");
                    mDataQueue.erase(mDataQueue.begin(), mDataQueue.begin() + 13);
                }
            }

            if ((int)data.find("AT+MWDISTANCE") == 0) {
                std::string distance = std::string(mDataQueue.begin() + 28, mDataQueue.begin() + 31);
                if (distance != "N/A" && isInt(distance)) {
                    Q_EMIT dataReceived(mTargetHost, "DISTANCE", std::stoi(distance));
                    mLogFile->write(", " + distance + "\n");
                } else {
                    // qDebug("DISTANCE: N/A");
                    Q_EMIT dataReceived(mTargetHost, "DISTANCE", -1);
                    mLogFile->write(", N/A\n");
                }
                mDataQueue.erase(mDataQueue.begin(), mDataQueue.begin() + 31);
            }
        }
    }

    void ComImpl::waitingRequestAuth() {
        if (mTargetHost != "" && !mAuthenticated) {
            this->sendData("hi", mTargetHost, (int) mTargetPort);
        }
    }

    void ComImpl::requestDataInfo() {
        if (mAuthenticated) {
            this->sendData("AT+MWRSSI\n", mTargetHost, (int)mTargetPort);
            this->sendData("AT+MWSNR\n", mTargetHost, (int)mTargetPort);
            this->sendData("AT+MWDISTANCE\n", mTargetHost, (int)mTargetPort);
        }
    }

    Com* Com::createInstance(Worker::ComRole comRole, Worker::SocketType sockType, uint16_t bindPort)
    {
        //if( !inst )
        //{
        //    inst = (Com* ) new ComImpl(comRole, sockType, bindPort);
        //}
        return (Com* ) new ComImpl(comRole, sockType, bindPort);
    }

    Com* Com::createInstance(Worker::ComRole comRole, const char *multicastAddress, uint16_t multicastPort)
    {
        //if( !inst )
        //{
        //    inst = (Com* ) new ComImpl(comRole, multicastAddress, multicastPort);
        //}
        return (Com* ) new ComImpl(comRole, multicastAddress, multicastPort);
    }

}
