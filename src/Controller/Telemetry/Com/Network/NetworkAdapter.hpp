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

#ifndef __NETWORK_ADAPTER_H_
#define __NETWORK_ADAPTER_H_

//================= Including C++ Libs =======================//
#include <string>
#include <vector>
#include <arpa/inet.h>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <errno.h>


//=============== Functions Lib API ======================//
namespace Worker {
    /**
     * Retrieve the host system's network hostname
     */
    std::string networkHostname();


    /**
     * Info about a particular network interface
     */
    struct networkAdapter_t
    {
        std::string name;
        std::string ipAddress;
    };

    /**
     * Retrieve info about the different network interfaces of the system.
     */
    void networkAdapters( std::vector<networkAdapter_t>& interfaceList );

}
#endif
