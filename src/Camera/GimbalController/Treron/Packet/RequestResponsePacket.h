#ifndef REQUESTRESPONSE_H
#define REQUESTRESPONSE_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye {
    class RequestResponsePacket
    {
    public:
    private:
        key_type m_packetID;
    public:
        RequestResponsePacket(){};

        RequestResponsePacket(const key_type _packetID)
        {
            m_packetID = _packetID;
        }
        inline key_type getPacketID() const
        {
            return m_packetID;
        }

        inline void setPacketID(const key_type &_packetID)
        {
            m_packetID = _packetID;
        }

        inline bool operator ==(const RequestResponsePacket& _requestResponsePacket)
        {
            return (m_packetID == _requestResponsePacket.m_packetID);
        }

        length_type size()
        {
            return sizeof(m_packetID);
        }

        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            _result.clear();
            _result = Utils::toByte<key_type>(m_packetID);
            return _result;
        }

        RequestResponsePacket* parse(std::vector<byte> _data, index_type _index = 0)
        {
            m_packetID = Utils::toValue<key_type>(_data, _index);
            return this;
        }

        RequestResponsePacket* parse(byte* _data, index_type _index = 0)
        {
            m_packetID = Utils::toValue<key_type>(_data, _index);
            return this;
        }
    };
}

#endif // REQUESTRESPONSE_H
