#ifndef IPENDPOINT_H
#define IPENDPOINT_H
#include <stdio.h>
#include <iostream>
using namespace std;
class IPEndPoint{
public:
    IPEndPoint(){
        ip = "0.0.0.0";
        port = 0;
    }
    IPEndPoint(string ip,unsigned short port){
        this->ip = ip;
        this->port = port;
    }
    ~IPEndPoint(){}
    string ip;
    unsigned short port;
};
#endif // IPENDPOINT_H
