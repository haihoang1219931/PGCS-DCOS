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
 
#include "NetworkAdapter.hpp"
#include "IPv4.hpp"

namespace Worker {
    // networkHostname
    std::string networkHostname()
    {
        char str[256];

        if( gethostname(str, sizeof(str)) != 0 )
            return "<error>";

        return str;
    }


    // networkAdapters
    void networkAdapters( std::vector<networkAdapter_t>& interfaceList )
    {
        struct ifaddrs* addrs;

        if( getifaddrs(&addrs) < 0 )
        {
            const int e = errno;
            const char* err = strerror(e);
            printf("Network error %i : %s\n", e, err );
        }

        for( ifaddrs* n=addrs; n != NULL; n = n->ifa_next )
        {
            if( n->ifa_addr->sa_family != AF_INET /*AF_INET6*/ )
                continue;

            if( !addrs->ifa_name || strlen(addrs->ifa_name) == 0 )
                continue;

            networkAdapter_t entry;

            entry.name      = addrs->ifa_name;
            entry.ipAddress = IPv4AddressStr(((sockaddr_in*)n->ifa_addr)->sin_addr.s_addr);

            printf("%s %s\n", entry.name.c_str(), entry.ipAddress.c_str());

            interfaceList.push_back(entry);
        }
    }
}

