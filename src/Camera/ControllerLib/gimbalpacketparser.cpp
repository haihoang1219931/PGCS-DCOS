#include "gimbalpacketparser.h"

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
//    printf("Data received: ");
//    for(int i = 0; i< length; i++){
//        printf(" 0x%02x",data[i]);
//    }
//    printf("\r\n");
}

void GimbalPacketParser::Parse(){
    while(receivedBuffer.size() > 0){
        if(receivedBuffer.at(0) != 0x24||receivedBuffer.at(1) != 0x40)
        {
            receivedBuffer.erase(receivedBuffer.begin(),receivedBuffer.begin() + 1);
            continue;
        }
        // packet length is not enough to parse
        if(receivedBuffer.size() < 6){
            receivedBuffer.clear();
            break;
        }
        key_type key = Utils::toValue<key_type>(receivedBuffer, 2);
        length_type dataLength = Utils::toValue<length_type>(receivedBuffer, 4);

        // packet size is wrong
        if(receivedBuffer.size() < dataLength + 7){
            receivedBuffer.clear();
            break;
        }

        byte checksum = KLV::calculateChecksum(receivedBuffer, 2, dataLength + 4);

        if(checksum == receivedBuffer.at(6 + dataLength)){
            vector<byte> data(receivedBuffer.begin() + 6, receivedBuffer.begin() + 6 + dataLength);
            receivedBuffer.erase(receivedBuffer.begin(), receivedBuffer.begin() + dataLength + 7);
//            printf("key = %d\ndata: ", key);
//            for(int i = 0; i < data.size(); i++){
//                printf("0x%02X ", data[i]);
//            }
//            printf("\n");
            Q_EMIT GimbalPacketParser::UavvGimbalPacketParsed(key, data);

        }else{
            receivedBuffer.erase(receivedBuffer.begin(), receivedBuffer.begin() + 2);
        }
    };
}

void GimbalPacketParser::Reset(){

}
