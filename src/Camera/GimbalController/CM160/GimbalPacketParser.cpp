#include "GimbalPacketParser.h"

GimbalPacketParser::GimbalPacketParser(QObject* parent) :
    QObject(parent)
{
    //receivedBuffer = vector<unsigned char>(1024);
}
GimbalPacketParser::~GimbalPacketParser(){

}

void GimbalPacketParser::Push(vector<unsigned char> data){

    receivedBuffer.insert(receivedBuffer.end(),data.begin(),data.end());
}

void GimbalPacketParser::Push(unsigned char* data, int length){
    for(int i = 0; i< length; i++){
        receivedBuffer.push_back(data[i]);
    }
}

void GimbalPacketParser::Parse(){
    while(receivedBuffer.size() > (unsigned int)MINIMUM_UAVV_PACKET_SIZE)
    {
        //qDebug("receivedBuffer has %d bytes",receivedBuffer.size());
        if (receivedBuffer[0] != (unsigned char)SyncBytes::Sync1)
        {
            receivedBuffer.erase(receivedBuffer.begin(), receivedBuffer.begin()+1);
            continue;
        }

        if(receivedBuffer[1] != (unsigned char)SyncBytes::Sync2)
        {
            receivedBuffer.erase(receivedBuffer.begin(), receivedBuffer.begin()+2);
            continue;
        }
        // CM160
        unsigned char length = receivedBuffer[2];
        unsigned char id = receivedBuffer[3];
        //printf("packet [%d-%d]\r\n",id,length);
        // Eye phoenix

        //unsigned char length = receivedBuffer[3];
        //unsigned char id = receivedBuffer[5];
        if(receivedBuffer.size() < (unsigned int)MINIMUM_UAVV_PACKET_SIZE + (unsigned int)length)
        {
            //printf("we dont have enough packets in the buffer yet, to parse the packet whose length we have found\r\n");
            //we dont have enough packets in the buffer yet, to parse the packet whose length we have found
            break;
        }

        // CM160
        vector<unsigned char> packetData(receivedBuffer.begin()+4,receivedBuffer.begin()+4+length);
        unsigned char packetChecksum = receivedBuffer[4 + length];

        // Eye phoenix

        //vector<unsigned char> packetData(receivedBuffer.begin()+6,receivedBuffer.begin()+6+length);
        //unsigned char packetChecksum = receivedBuffer[6 + length];
        /*
        qDebug("\r\n-------packetData---------\r\n");
        for(int i=0; i< packetData.size(); i++)
            qDebug(" %02x",packetData[i]);
        qDebug("\r\n--------------------------\r\n");

        qDebug("CalculateChecksum %08x vs %08x vs %08x",GimbalPacket::CalculateChecksum(receivedBuffer,3,1+length),receivedBuffer[3+1+length+1],packetChecksum);
        */
        // CM160
        if(GimbalPacket::CalculateChecksum(receivedBuffer,3,1+length) == packetChecksum)
        // Eye phoenix
        //if(GimbalPacket::CalculateChecksum(receivedBuffer,4,1+length) == packetChecksum)
        {
            GimbalPacket packet(id, packetData);
            //printf("GimbalPacket [%d]\r\n",id);
            Q_EMIT gimbalPacketParsed(packet, packetChecksum);
            // CM160
            receivedBuffer.erase(receivedBuffer.begin(),receivedBuffer.begin()+ 5 + length);
            // Eyephoenix
            //receivedBuffer.erase(receivedBuffer.begin(),receivedBuffer.begin()+ 7 + length);
        }
        else
        {
            // CM160
            receivedBuffer.erase(receivedBuffer.begin(), receivedBuffer.begin() + 2);
            // Eyephoenix
            //receivedBuffer.erase(receivedBuffer.begin(), receivedBuffer.begin() + 4);
        }
    }
}

void GimbalPacketParser::Reset(){

}

