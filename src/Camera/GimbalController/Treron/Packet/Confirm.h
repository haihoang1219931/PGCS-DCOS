#ifndef CONFIRM_H
#define CONFIRM_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
    class Confirm :public Object
    {
    public:
        Confirm(){

        }

        Confirm(const key_type &_key)
        {
            m_key = _key;
        }
        ~Confirm(){}
    public:
        inline void setKey(const key_type _key) {
            m_key = _key;
        }
        inline key_type getKey()const
        {
            return m_key;
        }

        length_type size()
        {
            return sizeof(m_key);
        }

        std::vector<byte> toByte()
        {
            std::vector<byte> _result = Utils::toByte<key_type>(m_key);
            return _result;
        }

        Confirm* parse(std::vector<byte> _data, index_type _index = 0)
        {
            byte* data = _data.data();
            m_key =  Utils::toValue<key_type>(data,_index);
            return this;
        }

        Confirm* parse(byte* _data, index_type _index = 0)
        {
            m_key =  Utils::toValue<key_type>(_data,_index);
            return this;
        }
    private:
        key_type m_key;
    };
}
#endif // CONFIRM_H
