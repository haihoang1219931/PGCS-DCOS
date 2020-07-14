#ifndef TARGETPOSITION_H
#define TARGETPOSITION_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"


namespace Eye
{
    class TargetPosition :public Object
    {
    public:
        TargetPosition(){}
        TargetPosition(const data_type _O_pn, const data_type _O_pe,const data_type _O_pd,
                       const data_type _A_pn, const data_type _A_pe,const data_type _A_pd,
                       const data_type _B_pn, const data_type _B_pe,const data_type _B_pd,
                       const data_type _C_pn, const data_type _C_pe,const data_type _C_pd,
                       const data_type _D_pn, const data_type _D_pe,const data_type _D_pd)
        {
            m_O_pn = _O_pn;
            m_O_pe = _O_pe;
            m_O_pd = _O_pd;

            m_A_pn = _A_pn;
            m_A_pe = _A_pe;
            m_A_pd = _A_pd;

            m_B_pn = _B_pn;
            m_B_pe = _B_pe;
            m_B_pd = _B_pd;

            m_C_pn = _C_pn;
            m_C_pe = _C_pe;
            m_C_pd = _C_pd;

            m_D_pn = _D_pn;
            m_D_pe = _D_pe;
            m_D_pd = _D_pd;
        }
        ~TargetPosition(){}
    public:
        //
        inline data_type getOpn()const{ return m_O_pn;}
        inline data_type getOpe()const{ return m_O_pe;}
        inline data_type getOpd()const{ return m_O_pd;}

        inline data_type getApn()const{ return m_A_pn;}
        inline data_type getApe()const{ return m_A_pe;}
        inline data_type getApd()const{ return m_A_pd;}

        inline data_type getBpn()const{ return m_B_pn;}
        inline data_type getBpe()const{ return m_B_pe;}
        inline data_type getBpd()const{ return m_B_pd;}

        inline data_type getCpn()const{ return m_C_pn;}
        inline data_type getCpe()const{ return m_C_pe;}
        inline data_type getCpd()const{ return m_C_pd;}

        inline data_type getDpn()const{ return m_D_pn;}
        inline data_type getDpe()const{ return m_D_pe;}
        inline data_type getDpd()const{ return m_D_pd;}

        inline void setOpn(const data_type _value){m_O_pn = _value;}
        inline void setOpe(const data_type _value){m_O_pe = _value;}
        inline void setOpd(const data_type _value){m_O_pd = _value;}

        inline void setApn(const data_type _value){m_A_pn = _value;}
        inline void setApe(const data_type _value){m_A_pe = _value;}
        inline void setApd(const data_type _value){m_A_pd = _value;}

        inline void setBpn(const data_type _value){m_B_pn = _value;}
        inline void setBpe(const data_type _value){m_B_pe = _value;}
        inline void setBpd(const data_type _value){m_B_pd = _value;}

        inline void setCpn(const data_type _value){m_C_pn = _value;}
        inline void setCpe(const data_type _value){m_C_pe = _value;}
        inline void setCpd(const data_type _value){m_C_pd = _value;}

        inline void setDpn(const data_type _value){m_D_pn = _value;}
        inline void setDpe(const data_type _value){m_D_pe = _value;}
        inline void setDpd(const data_type _value){m_D_pd = _value;}


        inline TargetPosition& operator =(const TargetPosition& _targetPosition)
        {
            m_O_pn = _targetPosition.m_O_pn;
            m_O_pe = _targetPosition.m_O_pe;
            m_O_pd = _targetPosition.m_O_pd;

            m_A_pn = _targetPosition.m_A_pn;
            m_A_pe = _targetPosition.m_A_pe;
            m_A_pd = _targetPosition.m_A_pd;

            m_B_pn = _targetPosition.m_B_pn;
            m_B_pe = _targetPosition.m_B_pe;
            m_B_pd = _targetPosition.m_B_pd;

            m_C_pn = _targetPosition.m_C_pn;
            m_C_pe = _targetPosition.m_C_pe;
            m_C_pd = _targetPosition.m_C_pd;

            m_D_pn = _targetPosition.m_D_pn;
            m_D_pe = _targetPosition.m_D_pe;
            m_D_pd = _targetPosition.m_D_pd;
            return *this;
        }

        inline bool operator ==(const TargetPosition& _targetPosition)
        {
            return (m_O_pn == _targetPosition.m_O_pn &&
                    m_O_pe == _targetPosition.m_O_pe &&
                    m_O_pd == _targetPosition.m_O_pd &&

                    m_A_pn == _targetPosition.m_A_pn &&
                    m_A_pe == _targetPosition.m_A_pe &&
                    m_A_pd == _targetPosition.m_A_pd &&

                    m_B_pn == _targetPosition.m_B_pn &&
                    m_B_pe == _targetPosition.m_B_pe &&
                    m_B_pd == _targetPosition.m_B_pd &&

                    m_C_pn == _targetPosition.m_C_pn &&
                    m_C_pe == _targetPosition.m_C_pe &&
                    m_C_pd == _targetPosition.m_C_pd &&

                    m_D_pn == _targetPosition.m_D_pn &&
                    m_D_pe == _targetPosition.m_D_pe &&
                    m_D_pd == _targetPosition.m_D_pd);
        }

        length_type size()
        {
            return sizeof(m_O_pn) * 10;
        }
        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            std::vector<byte> b_temp;

            _result = Utils::toByte<data_type>(m_O_pn);
            b_temp = Utils::toByte<data_type>(m_O_pe);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());
            b_temp = Utils::toByte<data_type>(m_O_pd);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());

            b_temp = Utils::toByte<data_type>(m_A_pn);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());
            b_temp = Utils::toByte<data_type>(m_A_pe);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());
            b_temp = Utils::toByte<data_type>(m_A_pd);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());

            b_temp = Utils::toByte<data_type>(m_B_pn);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());
            b_temp = Utils::toByte<data_type>(m_B_pe);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());
            b_temp = Utils::toByte<data_type>(m_B_pd);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());

            b_temp = Utils::toByte<data_type>(m_C_pn);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());
            b_temp = Utils::toByte<data_type>(m_C_pe);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());
            b_temp = Utils::toByte<data_type>(m_C_pd);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());

            b_temp = Utils::toByte<data_type>(m_D_pn);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());
            b_temp = Utils::toByte<data_type>(m_D_pe);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());
            b_temp = Utils::toByte<data_type>(m_D_pd);
            _result.insert(_result.end(),b_temp.begin(), b_temp.end());
            return _result;
        }
        TargetPosition* parse(std::vector<byte> _data, index_type _index = 0)
        {
            byte* data = _data.data();

            m_O_pn = Utils::toValue<data_type>(data, _index);
            m_O_pe = Utils::toValue<data_type>(data, _index + sizeof(data_type));
            m_O_pd = Utils::toValue<data_type>(data, _index + sizeof(data_type)*2);
//            printf("O: (%f,%f,%f)\r\n",m_O_pn,m_O_pe,m_O_pd);
            m_A_pn = Utils::toValue<data_type>(data, _index + sizeof(data_type)*3);
            m_A_pe = Utils::toValue<data_type>(data, _index + sizeof(data_type)*4);
            m_A_pd = Utils::toValue<data_type>(data, _index + sizeof(data_type)*5);
//            printf("A: (%f,%f,%f)\r\n",m_A_pn,m_A_pe,m_A_pd);
            m_B_pn = Utils::toValue<data_type>(data, _index + sizeof(data_type)*6);
            m_B_pe = Utils::toValue<data_type>(data, _index + sizeof(data_type)*7);
            m_B_pd = Utils::toValue<data_type>(data, _index + sizeof(data_type)*8);
//            printf("B: (%f,%f,%f)\r\n",m_B_pn,m_B_pe,m_B_pd);
            m_C_pn = Utils::toValue<data_type>(data, _index + sizeof(data_type)*9);
            m_C_pe = Utils::toValue<data_type>(data, _index + sizeof(data_type)*10);
            m_C_pd = Utils::toValue<data_type>(data, _index + sizeof(data_type)*11);
//            printf("C: (%f,%f,%f)\r\n",m_C_pn,m_C_pe,m_C_pd);
            m_D_pn = Utils::toValue<data_type>(data, _index + sizeof(data_type)*12);
            m_D_pe = Utils::toValue<data_type>(data, _index + sizeof(data_type)*13);
            m_D_pd = Utils::toValue<data_type>(data, _index + sizeof(data_type)*14);
//            printf("D: (%f,%f,%f)\r\n",m_D_pn,m_D_pe,m_D_pd);
            return this;
        }
        TargetPosition* parse(byte* _data, index_type _index = 0)
        {
            m_O_pn = Utils::toValue<data_type>(_data, _index);
            m_O_pe = Utils::toValue<data_type>(_data, _index + sizeof(data_type));
            m_O_pd = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*2);

            m_A_pn = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*3);
            m_A_pe = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*4);
            m_A_pd = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*5);

            m_B_pn = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*6);
            m_B_pe = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*7);
            m_B_pd = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*8);

            m_C_pn = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*9);
            m_C_pe = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*10);
            m_C_pd = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*11);

            m_D_pn = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*12);
            m_D_pe = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*13);
            m_D_pd = Utils::toValue<data_type>(_data, _index + sizeof(data_type)*14);

            return this;
        }
    private:
        data_type m_O_pn,m_O_pe,m_O_pd;
        data_type m_A_pn,m_A_pe,m_A_pd;
        data_type m_B_pn,m_B_pe,m_B_pd;
        data_type m_C_pn,m_C_pe,m_C_pd;
        data_type m_D_pn,m_D_pe,m_D_pd;
    };
}
#endif // TARGETPOSITION_H
