#ifndef UDPPAYLOAD_H
#define UDPPAYLOAD_H

#include <iostream>
#include <stdio.h>
#include <vector>
#include "IPEndpoint.h"
using namespace std;
class UdpPayload
{
public:
    UdpPayload();
    UdpPayload(vector<unsigned char> data, IPEndPoint source);
    ~UdpPayload();
    vector<unsigned char> Data;
    IPEndPoint Source;
};

#endif // UDPPAYLOAD_H
