#include "udppayload.h"

UdpPayload::UdpPayload()
{

}
UdpPayload::~UdpPayload()
{

}
UdpPayload::UdpPayload(vector<unsigned char> data, IPEndPoint source)
{
    Data = data;
    Source = source;
}
