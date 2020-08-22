#ifndef GIMBAL_CONTROL_H
#define GIMBAL_CONTROL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#define COMM_MAX_BUFF_SIZE  36
#define COMM_START_FRAME    0xA5

typedef enum _enum_control_gimbal_mode_t
{
    GIMBAL_OFF  = 0,
    GIMBAL_LOCK_MODE = 1,
    GIMBAL_FOLLOW_MODE = 2
}e_control_gimbal_mode;

int s16_Gimbal_Control_Init(void);
int s16_Gimbal_Control (int s16_roll_spd, int s16_tilt_spd, int s16_pan_spd);
int s16_Gimbal_Change_Mode(e_control_gimbal_mode mode);
//int copter_Send_Cmd(uint8_t* data, uint16_t dataSize);
#ifdef __cplusplus
}
#endif
#include <QtCore>
#include <QNetworkAccessManager>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

class GRGimbalController: public QObject
{
    Q_OBJECT
public:
    GRGimbalController();
    virtual~GRGimbalController();
    int setupTCP(QString address, int port);
    int set_control(int roll_speed, int tilt_speed, int pan_speed);
    int set_Mode(e_control_gimbal_mode mode);
    int getConnectionStatus();
private:
    int status;
    QTcpSocket * _socket;
    QString _ipAddress;
    int _ipPort;

    int16_t s16_Comm_Send_Message( uint16_t u16_msg_type, uint8_t *pu8_payload, uint16_t u16_len);
    uint32_t u32_CRC32 (uint32_t u32_crc, const uint8_t *pu8_data, uint32_t u32_count);
 public Q_SLOTS:
    void onConnected();
    void onDisconnected();

    void onTcpStateChanged(QAbstractSocket::SocketState socketState);
    void onTcpError(QAbstractSocket::SocketError error);

    void onReadyRead();
    void gimbalRetryConnect();
};

#endif

