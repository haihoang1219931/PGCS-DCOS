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

#ifndef SOCKETINPUTPARAMS_HPP
#define SOCKETINPUTPARAMS_HPP

#include "Network/Socket.hpp"
namespace Proxy {
    /**
     * @brief The SocketInputParams struct: Data container for input params setting up for socket
     */
    struct _socketInputParams {
        Worker::SocketType mSockType_ = Worker::SOCKET_UDP;
        std::string mAddress_ = "127.0.0.1";
        uint16_t mPort_ = 18802;
    };
    typedef _socketInputParams SocketInputParams;
}


#endif // SOCKETINPUTPARAMS_HPP
