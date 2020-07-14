#ifndef SENSORCONTROLLER_H
#define SENSORCONTROLLER_H

#include <QObject>
#include <QTcpSocket>
#include <QStringList>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
class SensorController : public QObject
{
    Q_OBJECT
public:
    explicit SensorController(QObject *parent = nullptr);
    virtual ~SensorController();
    int getConnectionStatus();
public:
    Q_INVOKABLE void sendRawData(QString command);
    Q_INVOKABLE void connectToHost(QString ip, int port);
    Q_INVOKABLE void disconnectFromHost();
Q_SIGNALS:
    void dataReceived(QString data);
    void dataSend(QString data);

public Q_SLOTS:
    void handlePacketReceived();
    void onConnected();
    void onDisconnected();

    void onTcpStateChanged(QAbstractSocket::SocketState socketState);
    void onTcpError(QAbstractSocket::SocketError error);
    void gimbalRetryConnect();
private:
    int m_status;
    QString m_ip;
    int m_port;
    QTcpSocket* m_socket = nullptr;
};

#endif // SENSORCONTROLLER_H
