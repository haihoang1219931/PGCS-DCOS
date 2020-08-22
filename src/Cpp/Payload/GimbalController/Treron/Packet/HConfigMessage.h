#ifndef HCONFIGMESSAGE_H
#define HCONFIGMESSAGE_H

#include <iostream>
#include <string>
#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye {
    class HConfigMessage
    {
    public:
        HConfigMessage(){
            m_messageSize = 0;
            m_funcCodeSize = 0;
        }
        HConfigMessage(std::string _funcCode, std::string _message)
            : m_message(_message), m_funcCode(_funcCode)
        {
            m_messageSize = (unsigned short)m_message.length();
            m_funcCodeSize = (unsigned short)m_funcCode.length();
        }

        ~HConfigMessage(){}

    private:
        std::string m_funcCode;
        std::string m_message;
        unsigned short m_messageSize;
        unsigned short m_funcCodeSize;

    public:
        void setFuncCode(std::string _funcCode){
            m_funcCode = _funcCode;
            m_funcCodeSize = _funcCode.size();
        }
        std::string getFuncCode(){ return m_funcCode; }

        void setMessage(std::string _message){
            m_message = _message;
            m_messageSize = m_message.size();
        }
        std::string getMessage(){ return m_message; }

        HConfigMessage& operator=(const HConfigMessage &_cm){
            m_funcCode = _cm.m_funcCode;
            m_message = _cm.m_message;
            m_messageSize = _cm.m_messageSize;
            m_funcCodeSize = _cm.m_funcCodeSize;
        }

        bool operator==(const HConfigMessage &_cm){
            return (m_funcCode == _cm.m_funcCode &&
                    m_message == _cm.m_message &&
                    m_messageSize == _cm.m_messageSize &&
                    m_funcCodeSize == _cm.m_funcCodeSize);
        }

        length_type size(){ return m_funcCode.length() + m_message.length() + sizeof(int); }

        std::vector<byte> toByte(){
            std::vector<byte> res, b_tmp;
            res = Utils::toByte<unsigned short>(m_funcCodeSize);
            b_tmp = std::vector<byte>(m_funcCode.begin(), m_funcCode.end());
            res.insert(res.end(), b_tmp.begin(), b_tmp.end());
            b_tmp = Utils::toByte<unsigned short>(m_messageSize);
            res.insert(res.end(), b_tmp.begin(), b_tmp.end());
            b_tmp = std::vector<byte>(m_message.begin(), m_message.end());
            res.insert(res.end(), b_tmp.begin(), b_tmp.end());
            return res;
        }

        HConfigMessage* parse(byte *_data, index_type _index = 0){
            m_funcCodeSize = Utils::toValue<unsigned short>(_data, _index);
            m_funcCode = std::string(_data + _index + sizeof(unsigned short),  _data + _index + sizeof(unsigned short) + m_funcCodeSize);
            m_messageSize = Utils::toValue<unsigned short>(_data, _index + sizeof(unsigned short) + m_funcCodeSize);
            m_message = std::string(_data + _index + 2 * sizeof(unsigned short) + m_funcCodeSize, _data + _index + 2 * sizeof(unsigned short) + m_funcCodeSize + m_messageSize);
            return this;
        }

        HConfigMessage* parse(std::vector<byte> data, index_type _index = 0){
            byte *_data = data.data();
            m_funcCodeSize = Utils::toValue<unsigned short>(_data, _index);
            m_funcCode = std::string(_data + _index + sizeof(unsigned short),  _data + _index + sizeof(unsigned short) + m_funcCodeSize);
            m_messageSize = Utils::toValue<unsigned short>(_data, _index + sizeof(unsigned short) + m_funcCodeSize);
            m_message = std::string(_data + _index + 2 * sizeof(unsigned short) + m_funcCodeSize, _data + _index + 2 * sizeof(unsigned short) + m_funcCodeSize + m_messageSize);
            return this;
        }
        static std::string makeGetCommand(const std::string &_funcCode, const int _dataSize){
            std::string res = "G";
            res.append(_funcCode);
            for(int i= 0; i < _dataSize; i++){ res.append("F"); }
            res.append("T");
            return res;
        }

        static std::string makeSetCommand(const std::string &_funcCode, const std::string _data){
            std::string res = "S";
            res.append(_funcCode);
            res.append(_data);
            res.append("T");
            return res;
        }

    };

}


#endif // HCONFIGMESSAGE_H
