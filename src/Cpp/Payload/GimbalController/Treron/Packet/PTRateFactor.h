#ifndef PT_RATE_FACTOR_H
#define PT_RATE_FACTOR_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

namespace Eye
{
    class PTRateFactor :public Object
    {
    public:
        PTRateFactor(){}
        PTRateFactor(const data_type &_id, const data_type &_panRateFactor, const data_type &_tiltRateFactor)
        {
            m_id = _id;
            m_panRateFactor = _panRateFactor;
            m_tiltRateFactor = _tiltRateFactor;
        }
        ~PTRateFactor(){}
    public:
        void setID(const index_type _id){
            m_id = _id;
        }

        index_type getID(){
            return m_id;
        }

        void setPanRateFactor(const data_type _panRateFactor) {
            m_panRateFactor = _panRateFactor;
        }
        data_type getPanRateFactor()const
        {
            return m_panRateFactor;
        }
        void setTiltRateFactor(const data_type _tiltRateFactor) {
            m_tiltRateFactor = _tiltRateFactor;
        }
        data_type getTiltRateFactor() const
        {
            return m_tiltRateFactor;
        }


        void setPTRateFactor(const index_type &_id, const data_type &_panRateFactor, const data_type &_tiltRateFactor)
        {
            m_id = _id;
            m_panRateFactor = _panRateFactor;
            m_tiltRateFactor = _tiltRateFactor;
        }
        length_type size()
        {
            return sizeof(m_id) + sizeof(m_panRateFactor) + sizeof(m_tiltRateFactor);
        }
        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            std::vector<byte> b_id, b_panRateFactor,b_tiltRateFactor;
            b_id = Utils::toByte<index_type>(m_id);
            b_panRateFactor = Utils::toByte<data_type>(m_panRateFactor);
            b_tiltRateFactor = Utils::toByte<data_type>(m_tiltRateFactor);

            _result.clear();
            _result.insert(_result.end(), b_id.begin(), b_id.end());
            _result.insert(_result.end(), b_panRateFactor.begin(), b_panRateFactor.end());
            _result.insert(_result.end(), b_tiltRateFactor.begin(), b_tiltRateFactor.end());
            return _result;
        }
        PTRateFactor* parse(std::vector<byte> _data, index_type _index = 0)
        {
            byte* data = _data.data();
            m_id = Utils::toValue<index_type>(data, _index);
            m_panRateFactor =  Utils::toValue<data_type>(data,_index + sizeof(index_type));
            m_tiltRateFactor = Utils::toValue<data_type>(data,_index + sizeof(index_type) + sizeof(m_tiltRateFactor));
            return this;
        }
        PTRateFactor* parse(byte* _data, index_type _index = 0)
        {
            m_id = Utils::toValue<index_type>(_data, _index);
            m_panRateFactor =  Utils::toValue<data_type>(_data,_index + sizeof(index_type));
            m_tiltRateFactor = Utils::toValue<data_type>(_data,_index + sizeof(index_type) + sizeof(m_tiltRateFactor));
            return this;
        }
    private:
        index_type m_id;
        data_type m_panRateFactor;
        data_type m_tiltRateFactor;
    };
}
#endif
