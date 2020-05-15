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
 
#include "IPv4.hpp"

namespace Worker {
    // IPv4Address
    bool IPv4Address( const char* str, uint32_t* ipOut )
    {
        if( !str || !ipOut )
            return false;

        in_addr addr;

        const int res = inet_pton(AF_INET, str, &addr);

        if( res != 1 )
        {
            printf("IPv4Address() - failed to convert '%s' to valid IPv4 address\n", str);
            return false;
        }

        *ipOut = addr.s_addr;
        return true;
    }


    // IPv4AddressStr
    std::string IPv4AddressStr( uint32_t ipAddress )
    {
        char str[INET_ADDRSTRLEN];
        memset(str, 0, INET_ADDRSTRLEN);

        if( inet_ntop(AF_INET, &ipAddress, str, INET_ADDRSTRLEN) == NULL )
            printf("IPv4AddressStr() - failed to convert 0x%08X to string\n", ipAddress);

        return std::string(str);
    }

}
