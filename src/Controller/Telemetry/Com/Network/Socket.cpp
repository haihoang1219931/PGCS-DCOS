/**
 * ===========================================================
 * Project: OnboardClient
 * Module: Network Communication
 * Module Short Description: UDP Socket Setup
 * Author: Trung Nguyen
 * Date: 10/30/2018
 * Viettel Aerospace Institude - Viettel Group
 * ===========================================================
 */

#include "Socket.hpp"

namespace Worker {

    // printErrno
    static void printErrno()
    {
        const int e = errno;
        const char* err = strerror(e);
        printf("Socket error %i : %s\n", e, err);
    }

    // constructor
    Socket::Socket( SocketType type ) : mType(type)
    {
        mSock       = -1;
        mLocalPort  = 0;
        mLocalIP 	= 0;
        mRemoteIP   = 0;
        mRemotePort = 0;

        mListening        = false;
        mPktInfoEnabled   = false;
        mBroadcastEnabled = false;
        mMulticastEnabled = true;
        mTcpConnected     = false;
    }


    // destructor
    Socket::~Socket()
    {
        if( mSock != -1 )
        {
            close(mSock);
            mSock = -1;
        }
    }


    // SetBufferSize
    bool Socket::SetBufferSize( size_t size )
    {
        if( size == 0 )
            return false;

        const int isz = size;

        if( setsockopt(mSock, SOL_SOCKET, SO_RCVBUF, &isz, sizeof(int)) != 0 )
        {
            printf("Socket failed to set rx buffer size of %zu bytes.\n", size);
            printErrno();
            return false;
        }

        if( setsockopt(mSock, SOL_SOCKET, SO_SNDBUF, &isz, sizeof(int)) != 0 )
        {
            printf("Socket failed to set rx buffer size of %zu bytes.\n", size);
            printErrno();
            return false;
        }

        printf("successfully set socket buffer size of %s:%u to %zu bytes\n", IPv4AddressStr(mLocalIP).c_str(), (uint32_t)mLocalPort, size);
        return true;
    }


    bool Socket::EnableJumboBuffer()
    {
        return SetBufferSize(167772160); //SetBufferSize(10485760);
    }


    // Create
    Socket* Socket::Create( SocketType type )
    {
        Socket* s = new Socket(type);

        if( !s )
            return NULL;

        if( !s->Init() )
        {
            delete s;
            return NULL;
        }

        return s;
    }


    // Init
    bool Socket::Init()
    {
        const int type = (mType == SOCKET_UDP) ? SOCK_DGRAM : SOCK_STREAM;

        if( (mSock = socket(AF_INET, type, 0)) < 0 )
        {
            printErrno();
            return false;
        }

        return true;
    }


    // Accept
    bool Socket::Accept( uint64_t timeout )
    {
        if( mType != SOCKET_TCP )
            return true;

        // set listen mode
        if( !mListening )
        {
            if( listen(mSock, 1) < 0 )
            {
                printf("failed to listen() on socket.\n");
                return false;
            }

            mListening = true;
        }

        // select timeout
        if( timeout != 0 )
        {
            struct timeval tv;

            tv.tv_sec  = timeout / 1000000;
            tv.tv_usec = timeout - (tv.tv_sec * 1000000);

            fd_set fds;

            FD_ZERO(&fds);
            FD_SET(mSock, &fds);

            const int result = select(mSock + 1, &fds, NULL, NULL, &tv);

            if( result < 0 )
            {
                printf("select() error occurred during Socket::Accept()   (code=%i)\n", result);
                printErrno();
                return false;
            }
            else if( result == 0 )
            {
                printf("Socket::Accept() timeout occurred\n");
                return false;
            }
        }

        // accept connections
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        socklen_t addrLen = sizeof(addr);

        const int fd = accept(mSock, (struct sockaddr*)&addr, &addrLen);

        if( fd < 0 )
        {
            printf("Socket::Accept() failed  (code=%i)\n", fd);
            printErrno();
            return false;
        }

        mRemoteIP   = addr.sin_addr.s_addr;
        mRemotePort = ntohs(addr.sin_port);

        // swap out the old 'listening' port
        close(mSock);

        mSock      = fd;
        mListening = false;

        return true;
    }


    // Bind
    bool Socket::Bind( const char* ipStr, uint16_t port )
    {
        if( !ipStr )
            return Bind(port);

        uint32_t ipAddress = 0;

        if( !IPv4Address(ipStr, &ipAddress) )
            return false;

        return Bind(ipAddress, port);
    }


    // Bind
    bool Socket::Bind( uint32_t ipAddress, uint16_t port )
    {
        // If multicast socket method enable => allow multiple applications to receive datagrams that are destined to the same local port number
        if( mMulticastEnabled )
        {
            int reuse = 1;
            if( setsockopt( mSock, SOL_SOCKET, SO_REUSEADDR, ( char * )&reuse, sizeof( reuse ) ) < 0 )
            {
                printf("failed to set SO_REUSEADDR error.! \n");
                printErrno();
                return false;
            }
            printf( "Setting SO_REUSEADDR ok...! \n");
        }

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));

        addr.sin_family 	 = AF_INET;
        addr.sin_addr.s_addr = ipAddress;
        addr.sin_port		 = htons(port);

        if( bind(mSock, (struct sockaddr*)&addr, sizeof(addr)) < 0 )
        {
            printf("failed to bind socket to %s port %hu\n", IPv4AddressStr(ipAddress).c_str(), port);
            printErrno();
            return false;
        }

        mLocalIP   = ipAddress;
        mLocalPort = port;


        return true;
    }


    // Bind
    bool Socket::Bind( uint16_t port )
    {
        return Bind( htonl(INADDR_ANY), port );
    }

    // Join multicast
    bool Socket::joinMulticast(const char *multicastGroup)
    {
        struct ip_mreq mreq;
        uint32_t ipAddress = 0;
        if( !IPv4Address( multicastGroup, &ipAddress ) )
            return false;

        mreq.imr_multiaddr.s_addr = ipAddress;
        mreq.imr_interface.s_addr = htonl( INADDR_ANY );
        if( setsockopt( mSock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof( mreq ) ) < 0 )
        {
            printf("failed to join socket group. ! \n");
            return false;
        }
        return true;
    }

    // Connect
    bool Socket::Connect( uint32_t ipAddress, uint16_t port )
    {
        if( mType != SOCKET_TCP || mSock == -1)
            return false;

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));

        addr.sin_family 	 = AF_INET;
        addr.sin_addr.s_addr = ipAddress;
        addr.sin_port		 = htons(port);

        if( !mTcpConnected )
        {
            if( ::connect(mSock, (struct sockaddr*)&addr, sizeof(addr)) < 0 )
            {
                printf("Socket failed to connect to %X port %hi.\n", ipAddress, port);
                printErrno();
                return false;
            }  else
            {
                mTcpConnected = true;
            }
        }

        mRemoteIP = ipAddress;
        mRemotePort = port;

        return true;
    }


    // Connect
    bool Socket::Connect( const char* ipStr, uint16_t port )
    {
        if( !ipStr || mType != SOCKET_TCP )
            return false;

        uint32_t ipAddress = 0;

        if( !IPv4Address(ipStr, &ipAddress) )
            return false;

        return Connect(ipAddress, port);
    }


    // Recieve
    size_t Socket::Recieve( uint8_t* buffer, size_t size, uint32_t* srcIpAddress, uint16_t* srcPort )
    {
        if( !buffer || size == 0 )
            return 0;

        struct sockaddr_in srcAddr;
        socklen_t addrLen = sizeof(srcAddr);

        // recieve packet
        const int64_t res = recvfrom(mSock, (void*)buffer, size, 0, (struct sockaddr*)&srcAddr, &addrLen);
        if( res < 0 )
        {
            //printf(LOG_SYS "Socket::Recieve() timed out\n");
            //printErrno();
            return 0;
        }

        if( srcIpAddress != NULL )
            *srcIpAddress = srcAddr.sin_addr.s_addr;

        if( srcPort != NULL )
            *srcPort = ntohs(srcAddr.sin_port);

    #if 0	// DEBUG
        printf(LOG_SYS "recieved %04lli bytes from %s:%hu\n", res, IPv4AddressStr(srcAddr.sin_addr.s_addr).c_str(), (uint16_t)ntohs(srcAddr.sin_port));
    #endif

        return res;
    }


    // Recieve
    size_t Socket::Recieve( char* buffer, size_t size, uint32_t* remoteIP, uint16_t* remotePort, uint32_t* localIP )
    {
        if( !buffer || size == 0 )
            return 0;


        // enable IP_PKTINFO if not already done so
        if( !mPktInfoEnabled )
        {
            int opt = 1;

            if( setsockopt(mSock, IPPROTO_IP, IP_PKTINFO, (const char*)&opt, sizeof(int)) != 0 )
            {
                printf("Socket::Receive() failed to enabled extended PKTINFO\n");
                printErrno();
                return 0;
            }

            mPktInfoEnabled = true;
        }


        // setup msghdr to recieve addition address info
        union controlData {
            cmsghdr cmsg;
            uint8_t data[CMSG_SPACE(sizeof(struct in_pktinfo))];
        };

        iovec iov;
        msghdr msg;
        controlData cmsg;
        sockaddr_in remoteAddr;

        memset(&msg, 0, sizeof(msghdr));
        memset(&cmsg, 0, sizeof(cmsg));
        memset(&remoteAddr, 0, sizeof(sockaddr_in));

        iov.iov_base = buffer;
        iov.iov_len  = size;

        msg.msg_name    = &remoteAddr;
        msg.msg_namelen = sizeof(sockaddr_in);
        msg.msg_iov     = &iov;
        msg.msg_iovlen  = 1;
        msg.msg_control = &cmsg;
        msg.msg_controllen = sizeof(cmsg);


        // recieve message
        const ssize_t res = recvmsg(mSock, &msg, 0);

        if( res < 0 )
        {
            //printf(LOG_SYS "Socket::Recieve() timed out\n");
            //printErrno();
            return 0;
        }

        // output local address
        for( cmsghdr* c = CMSG_FIRSTHDR(&msg); c != NULL; c = CMSG_NXTHDR(&msg, c) )
        {
            if( c->cmsg_level == IPPROTO_IP && c->cmsg_type == IP_PKTINFO )
            {
                if( localIP != NULL )
                    *localIP = ((in_pktinfo*)CMSG_DATA(c))->ipi_addr.s_addr;

                // TODO local port...not included in IP_PKTINFO?
            }
        }

        // output remote address
        if( remoteIP != NULL )
            *remoteIP = remoteAddr.sin_addr.s_addr;

        if( remotePort != NULL )
            *remotePort = ntohs(remoteAddr.sin_port);

        return res;
    }


    // SetRecieveTimeout
    bool Socket::SetRecieveTimeout( uint64_t timeout )
    {
        struct timeval tv;

        tv.tv_sec  = timeout / 1000000;
        tv.tv_usec = timeout - (tv.tv_sec * 1000000);

        if( setsockopt(mSock, SOL_SOCKET, SO_RCVTIMEO, (char*)&tv, sizeof(struct timeval)) != 0 )
        {
            printf("Socket::SetRecieveTimeout() failed to set timeout of %zu microseconds.\n", timeout);
            printErrno();
            return false;
        }

        return true;
    }


    // Send
    bool Socket::Send( const void* buffer, size_t size, uint32_t remoteIP, uint16_t remotePort )
    {
        if( !buffer || size == 0 ) {
            return false;
        }


        // if sending broadcast, enable broadcasting if not already done so
        if( remoteIP == netswap32(IP_BROADCAST) && !mBroadcastEnabled )
        {
            int opt = 1;

            if( setsockopt(mSock, SOL_SOCKET, SO_BROADCAST, (const char*)&opt, sizeof(int)) != 0 )
            {
                printf("Socket::Send() failed to enabled broadcasting...\n");
                printErrno();
                return false;
            }

            mBroadcastEnabled = true;
        }

        // send the message
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));

        addr.sin_family 	 = AF_INET;
        addr.sin_addr.s_addr = remoteIP;
        addr.sin_port		 = htons(remotePort);
        const int64_t res = sendto(mSock, (void*)buffer, size, 0, (struct sockaddr*)&addr, sizeof(addr));
        if( res != size )
        {
            printf("failed send() to %s port %hu  (%li of %zu bytes)\n", IPv4AddressStr(remoteIP).c_str(), remotePort, res, size);
            printErrno();
            return false;
        }
        return true;
    }


    // PrintIP
    void Socket::PrintIP() const
    {
        printf("Socket %i   host %s:%hu   remote %s:%hu\n", mSock, IPv4AddressStr(mLocalIP).c_str(), mLocalPort,
                                                              IPv4AddressStr(mRemoteIP).c_str(), mRemotePort );
    }


    // GetMTU
    size_t Socket::GetMTU()
    {
        int mtu = 0;
        socklen_t mtuSize = sizeof(int);

        if( getsockopt(mSock, IPPROTO_IP, IP_MTU, &mtu, &mtuSize) < 0 )
        {
            printf("Socket::GetMTU() -- getsockopt(SOL_IP, IP_MTU) failed.\n");
            printErrno();
            return 0;
        }

        return (size_t)mtu;
    }

    // Enable multicast
    void Socket::setEnableMulticast(bool enable)
    {
        mMulticastEnabled = enable;
    }
}
