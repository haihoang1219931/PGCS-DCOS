#ifndef TELEMETRYCONTROLLER_H
#define TELEMETRYCONTROLLER_H

#include <QObject>
#include <QTcpSocket>
#include <QStringList>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <QTimer>
#include <QRegularExpression>
class TelemetryController : public QObject
{
    Q_OBJECT
public:
    explicit TelemetryController(QObject *parent = nullptr);
    virtual ~TelemetryController();
    int getConnectionStatus();
public:
    Q_INVOKABLE void sendRawData(QString command);
    Q_INVOKABLE void connectToHost(QString ip, int port);
    Q_INVOKABLE void disconnectFromHost();
    void parseData(QString data);
Q_SIGNALS:
    void dataReceived(QString data);
    void dataSend(QString data);
    void writeLogTimeout(float snr, float rssi);
public Q_SLOTS:
    void handlePacketReceived();
    void onConnected();
    void onDisconnected();

    void onTcpStateChanged(QAbstractSocket::SocketState socketState);
    void onTcpError(QAbstractSocket::SocketError error);
    void gimbalRetryConnect();
    void handleRequestTimeout();
    void handleWriteLogTimeout();
private:
    int m_status;
    QString m_ip;
    int m_port;
    QTcpSocket* m_socket = nullptr;
    QTimer* m_timerRequest = nullptr;
    QTimer* m_timerWriteLog = nullptr;
    int m_countRequest = 0;
    bool m_sendUser = false;
    bool m_sendPass = false;
    bool m_authenticated = false;
    QByteArray m_buffer;
    int m_snr = -1;
    int m_rssi = -1;
    int m_distance = -1;
};

#endif // TELEMETRYCONTROLLER_H
