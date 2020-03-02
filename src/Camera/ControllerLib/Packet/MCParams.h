#ifndef MCPARAMS_H
#define MCPARAMS_H


#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
    class MCParams :public Object
    {
    public:
        MCParams(){

        }

        MCParams(const data_type &_kp, const data_type &_ki)
        {
            m_kp = _kp;
            m_ki = _ki;
        }
        ~MCParams(){}

    private:
        data_type m_kp;
        data_type m_ki;

    public:
        inline void setKp(const data_type _kp) {
            m_kp = _kp;
        }

        inline data_type getKp()const
        {
            return m_kp;
        }

        inline void setKi(const data_type _ki) {
            m_ki = _ki;
        }

        inline data_type getKi()const
        {
            return m_ki;
        }

        length_type size()
        {
            return 2 * sizeof(data_type);
        }

        std::vector<byte> toByte()
        {
            std::vector<byte> _result = Utils::toByte<data_type>(m_kp);
            std::vector<byte> b_ki = Utils::toByte<data_type>(m_ki);
            _result.insert(_result.end(), b_ki.begin(), b_ki.end());
            return _result;
        }

        MCParams* parse(std::vector<byte> _data, index_type _index = 0)
        {
            byte* data = _data.data();
            m_kp =  Utils::toValue<data_type>(data,_index);
            m_ki = Utils::toValue<data_type>(data, _index + sizeof(data));
            return this;
        }

        MCParams* parse(byte* _data, index_type _index = 0)
        {
            m_kp =  Utils::toValue<data_type>(_data,_index);
            m_ki = Utils::toValue<data_type>(_data, _index + sizeof(data_type));
            return this;
        }
    };
}


#endif // MCPARAMS_H
