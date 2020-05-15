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


#ifndef __NETWORK_IPV4_H_
#define __NETWORK_IPV4_H_

//============= Including C++ Libs =========================//
#include <string>
#include <arpa/inet.h>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <errno.h>

//============= Functions Lib API =========================//
namespace Worker {
    /**
     * Convert an IPv4 address string in "xxx.xxx.xxx.xxx" format to binary representation.
     * @param str the IPv4 string, in "xxx.xxx.xxx.xxx" format
     * @param ip_out output pointer to converted IPv4 address, in network byte order.
     * @returns true, if str was a valid IPv4 address and the conversion was successful.
     *          false, if the conversion failed.
     */
    bool IPv4Address( const char* str, uint32_t* ip_out );


    /**
     * Return text string of IPv4 address in "xxx.xxx.xxx.xxx" format
     * @param ip_address IPv4 address, supplied in network byte order.
     */
    std::string IPv4AddressStr( uint32_t ip_address );

}

#endif
