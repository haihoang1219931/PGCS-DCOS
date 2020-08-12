#include "BufferOut.h"

BufferOut::BufferOut(QObject *parent) : QObject(parent)
{

}
BufferOut::~BufferOut()
{
    uinit();
}
void BufferOut::setIP(string ip){
    m_ip = ip;
}

void BufferOut::setPort(int port){
    m_port = port;
}

void BufferOut::init(){
    m_udpSocket = socket(PF_INET, SOCK_DGRAM, 0);
    m_udpAddress.sin_family = AF_INET;
    m_udpAddress.sin_port = htons(m_port);
    m_udpAddress.sin_addr.s_addr = inet_addr((const char*)(m_ip.c_str()));
    isInitialized = true;
}

void BufferOut::uinit(){
    if(isInitialized == true){
        close(m_udpSocket);
        isInitialized = false;
    }
}

void BufferOut::add(vector<unsigned char> data){
    m_dataSend.insert(m_dataSend.begin(),data.begin(),data.end());
}

void BufferOut::send(){
    while(m_dataSend.size() >= 7){
        int numByteSend = sendto(m_udpSocket,m_dataSend.data(),m_dataSend.size(),0,(struct sockaddr *)&m_udpAddress,sizeof(m_udpAddress));
        if(numByteSend > 0){
            m_dataSend.erase(m_dataSend.begin(),m_dataSend.begin()+numByteSend);
        }else{
            break;
        }
    }
}

void BufferOut::send(vector<unsigned char> data){
    sendto(m_udpSocket,data.data(),data.size(),0,(struct sockaddr *)&m_udpAddress,sizeof(m_udpAddress));
}
