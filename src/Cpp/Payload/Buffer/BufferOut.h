#ifndef BUFFEROUT_H
#define BUFFEROUT_H

#ifdef _WIN32
    #include <winsock.h>
    #include <windows.h>
    #include <time.h>
    #define PORT        unsigned long
    #define ADDRPOINTER   int*
    struct _INIT_W32DATA
    {
       WSADATA w;
       _INIT_W32DATA() { WSAStartup( MAKEWORD( 2, 1 ), &w ); }
    } _init_once;
#else       /* ! win32 */
    #include <unistd.h>
    #include <sys/time.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netdb.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #define PORT        unsigned short
    #define SOCKET    int
    #define HOSTENT  struct hostent
    #define SOCKADDR    struct sockaddr
    #define SOCKADDR_IN  struct sockaddr_in
    #define ADDRPOINTER  unsigned int*
    #define INVALID_SOCKET -1
    #define SOCKET_ERROR   -1
#endif /* _WIN32 */

#include <iostream>
#include <vector>
#include <QObject>
#include "../Packet/Common_type.h"
using namespace std;
class BufferOut : public QObject
{
    Q_OBJECT
public:
    explicit BufferOut(QObject *parent = 0);
    virtual ~BufferOut();
Q_SIGNALS:

public Q_SLOTS:
    void send();
public:
    void connectToHost(QString ip, int port);
    void setIP(string ip);
    void setPort(int port);
    void init();
    void uinit();
    void add(vector<unsigned char> data);
    void send(vector<unsigned char> data);
public:
    vector<unsigned char> m_dataSend;
    bool isInitialized = false;
    SOCKET m_udpSocket;
    SOCKADDR_IN m_udpAddress;
    std::string m_ip;
    int m_port;
};

#endif // BUFFEROUT_H
