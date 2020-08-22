#include "UavvPacket.h"
GimbalPacket::GimbalPacket()
{

}
GimbalPacket::GimbalPacket(unsigned char idByte, unsigned char *data,int dataLength){
    IDByte = idByte;
    for(int i=0; i< dataLength; i++){
        Data.push_back(data[i]);
    }
}

GimbalPacket::GimbalPacket(unsigned char idByte, vector<unsigned char> data){
    IDByte = idByte;
    Data = data;
}
GimbalPacket::GimbalPacket(UavvGimbalProtocol id, unsigned char *data,int dataLength){
    IDByte = (unsigned char)id;
    for(int i=0; i< dataLength; i++){
        Data.push_back(data[i]);
    }
}

GimbalPacket::GimbalPacket(UavvGimbalProtocol id, vector<unsigned char> data){
    IDByte = (unsigned char)id;
    Data = data;
}
vector<unsigned char> GimbalPacket::encode(){
    //printf("Encoding data\r\n");
    vector <unsigned char> buffer;
    buffer.push_back(Sync1);
    buffer.push_back(Sync2);
    // Eyephoenix
    //buffer.push_back((unsigned char)0x00);
    buffer.push_back((unsigned char)Data.size());
    // Eyephoenix
    //buffer.push_back((unsigned char)0x00);
    buffer.push_back(IDByte);
    //printf("encoded %d byte\r\n",buffer.size());
    buffer.insert(buffer.end(),Data.begin(),Data.end());
    // CM160
    unsigned char checksum = GimbalPacket::CalculateChecksum(buffer,3, 1 + Data.size());
    // Eyephoenix
    //unsigned char checksum = GimbalPacket::CalculateChecksum(buffer,4, 2 + Data.size());
    buffer.push_back(checksum);
    /*
    printf("Data send:");
    for(int i=0;i<buffer.size(); i++){
        printf("0x%02x ",buffer[i]);
    }
    printf("\r\n");
    */
    return buffer;
}

unsigned char GimbalPacket::CalculateChecksum(vector<unsigned char> data, int startIndex, int length){
    short csum = 0;
    for (int i = 0; i < length; i++)
    {
        csum += (short)data[i+startIndex];
    }
    csum = (255 - (csum % 255));
    return (unsigned char)csum;
}
