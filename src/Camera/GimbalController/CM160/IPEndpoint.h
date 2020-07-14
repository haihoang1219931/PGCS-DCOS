#ifndef IPENDPOINT_H
#define IPENDPOINT_H
#include <QObject>
#include <stdio.h>
#include <iostream>
using namespace std;
class IPEndPoint{
public:
    IPEndPoint(){
        ip = "0.0.0.0";
        port = 0;
    }
    IPEndPoint(QString ip,int port){
        this->ip = ip.toStdString();
        this->port = (unsigned short)port;
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
