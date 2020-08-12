#ifndef KLV_H
#define KLV_H

#include "Object.h"
#include "utils.h"

const unsigned char Sync1 = 0x24;
const unsigned char Sync2 = 0x40;

namespace Eye
{
    enum ParseResult
    {
        Success = 0,
        InvalidKLV = 1,
        InvalidLength = 2,
        InvalidData = 3,
        InvalidId = 4,
        Error = 5,
        InvalidObject = 6,
    };

    class KLV
    {
    public:
        key_type key;
        std::vector<byte> data; //only data of KLV
        static const int MINIMUM_UAVV_PACKET_SIZE = 7;
    public:
        KLV()
        {

        }
        KLV(key_type _key, byte * _data, length_type _len)
        {
            key = _key;
            data = std::vector<byte>(_data, _data + _len);
        }

        KLV(key_type _key, std::vector<byte> _data)
        {
            key = _key;
            data = _data;
        }

        ~KLV()
        {

        }
    public:
        std::vector<byte> encode(){
            std::vector<byte> buffer;
            buffer.push_back(Sync1);
            buffer.push_back(Sync2);

            std::vector<byte> s_key;
            s_key = Utils::toByte<key_type>(key);
            buffer.insert(buffer.end(), s_key.begin(), s_key.end());

            std::vector<byte> s_len;
            s_len = Utils::toByte<length_type>(data.size());
            buffer.insert(buffer.end(), s_len.begin(), s_len.end());

            buffer.insert(buffer.end(), data.begin(), data.end());
            byte checksum = calculateChecksum(buffer, 2, data.size() + 4);
            buffer.push_back(checksum);
            return buffer;
        }

        std::vector<byte> encode(std::pair<key_type, std::vector<byte>> &_item){
            key = _item.first;
            data = _item.second;
            return encode();
        }

        std::vector<byte> encode(key_type _key, Object *_obj)
        {
            key = _key;
            data = _obj->toByte();

            std::vector<byte> buffer;
            buffer.push_back(Sync1);
            buffer.push_back(Sync2);

            std::vector<byte> s_key;
            s_key = Utils::toByte<key_type>(_key);
            buffer.insert(buffer.end(), s_key.begin(), s_key.end());

            std::vector<byte> s_len;
            s_len = Utils::toByte<length_type>(_obj->size());
            buffer.insert(buffer.end(), s_len.begin(), s_len.end());

            buffer.insert(buffer.end(), data.begin(), data.end());
            byte checksum = calculateChecksum(data, 0, _obj->size());
            buffer.push_back(checksum);
            return buffer;
        }
        std::vector<byte> encode(key_type _key, byte * _data, length_type _len)
        {
            key = _key;
            data.clear();
            data = std::vector<byte>(_data, _data + _len);

            std::vector<byte> buffer;
            buffer.push_back(Sync1);
            buffer.push_back(Sync2);

            std::vector<byte> s_key;
            s_key = Utils::toByte<key_type>(_key);
            buffer.insert(buffer.end(), s_key.begin(), s_key.end());

            std::vector<byte> s_len;
            s_len = Utils::toByte<length_type>(_len);
            buffer.insert(buffer.end(), s_len.begin(), s_len.end());

            buffer.insert(buffer.end(), data.begin(), data.end());
            byte checksum = calculateChecksum(data, 0, _len);
            buffer.push_back(checksum);
            return buffer;

        }
        ParseResult tryParse(byte* _data, index_type startIndex, int _dataSize)
        {
            index_type index;
            index = startIndex;
            if (_data[index] != Sync1 || _data[index + 1] != Sync2)
            {
                return ParseResult::InvalidKLV;
            }

            key = Utils::toValue<key_type>(_data, index + 2);
            length_type _length = Utils::toValue<length_type>(_data, index + 4);
            if(_length > _dataSize - 5){
                return ParseResult::InvalidLength;
            }
            unsigned char _checksum = calculateChecksum(_data, index + 2, _length + 4);
            if (_checksum != _data[index + _length + 6])
            {
                return ParseResult::Error;
            }

            data = std::vector<unsigned char>(_data + index + 6, _data + index + _length + 6);
            startIndex = index + _length + 6;
            return ParseResult::Success;
        }
        ParseResult tryParse(std::vector<byte> _data)
        {
            if(_data.size() <= 0) return ParseResult::InvalidLength;
            index_type index = 0;
            if (_data[index] != Sync1 || _data[index + 1] != Sync2)
            {
                return ParseResult::InvalidKLV;
            }
            key = Utils::toValue<key_type>(_data, index + 2);
            length_type _length = Utils::toValue<length_type>(_data, index + 4);
            if(_length > _data.size() - 5){
                return ParseResult::InvalidLength;
            }
            unsigned char _checksum = calculateChecksum(_data, index + 2, _length + 4);

            if (_checksum != _data.back())
            {
                return ParseResult::Error;
            }
            data.clear();
            data.insert(data.begin(), _data.begin() + 6, _data.begin() + 6 + _length);
            return ParseResult::Success;
        }
        static unsigned char calculateChecksum(std::vector<unsigned char> data, index_type startIndex, length_type length)
        {
            index_type csum = 0;
            //printf("caculating checksum:\r\n");
            for (index_type i = 0; i < length; i++)
            {

                csum += (short)data[i + startIndex];
                //printf("0x%02x ",(short)data[i + startIndex]);
            }
            csum = (255 - (csum % 255));
            //printf("\r\n");
            return (unsigned char)csum;
        }
        static unsigned char calculateChecksum(unsigned char* data, index_type startIndex, length_type length)
        {
            index_type csum = 0;
            for (index_type i = 0; i < length; i++)
            {
                csum += (short)data[i + startIndex];
            }
            csum = (255 - (csum % 255));
            return (unsigned char)csum;
        }

//        decode the bytes into individual packets
//        return number of packets
//        static int decode(std::vector<byte> &bytes, std::vector<std::pair<key_type, std::vector<byte>>> &packets){
//            int res = -1;
//        }

    private:
        //static std::map<key_type, Object*> m_map_object;
    };
};
#endif
