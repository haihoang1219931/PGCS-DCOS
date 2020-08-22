#ifndef IMAGE_STAB_H
#define IMAGE_STAB_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"
#include "EyeStatus.h"

using namespace Status;
namespace Eye
{
    class ImageStab:public Object
    {
    public:
        ImageStab(){
            m_mode = (byte) VideoStabMode::ON;
            m_cropRatio = 0.0;
        }

        ImageStab(byte _mode, data_type _crop) :m_mode(_mode), m_cropRatio(_crop)
        {
        }

        ~ImageStab()
        {
        }

        inline ImageStab & operator=(const ImageStab &_mode)
        {
            m_mode = _mode.m_mode;
            m_cropRatio = _mode.m_cropRatio;
            return *this;
        }

        inline bool operator==(const ImageStab &_mode)
        {
            return (m_mode == _mode.m_mode&&m_cropRatio == _mode.m_cropRatio);
        }

        inline void setStabMode(const byte _mode)
        {
            m_mode = _mode;
        }

        inline void setCropRatio(const data_type _crop)
        {
            m_cropRatio = _crop;
        }

        inline byte getStabMode()const
        {
            return m_mode;
        }

        inline data_type getCropRatio()const
        {
            return m_cropRatio;
        }

        inline void setImageStab(const byte _mode, const data_type _crop)
        {
            m_mode = _mode;
            m_cropRatio = _crop;
        }

        length_type size()
        {
            return sizeof(m_mode) + sizeof(m_cropRatio) ;
        }

        std::vector<byte> toByte()
        {
            std::vector<byte> _result, b_tmp;
            _result.push_back(m_mode);
            b_tmp = Utils::toByte<data_type>(m_cropRatio);
            _result.insert(_result.end(), b_tmp.begin(), b_tmp.end());
            return _result;
        }

        ImageStab* parse(std::vector<byte> _data, index_type _index = 0)
        {
            byte * data = _data.data();
            m_mode = data[_index];
            m_cropRatio = Utils::toValue<data_type>(data, _index + sizeof(m_mode));
            return this;
        }

        ImageStab* parse(byte* _data, index_type _index = 0)
        {
            m_mode = _data[_index];
            m_cropRatio = Utils::toValue<data_type>(_data, _index + sizeof(m_mode));
            return this;
        }
    private:
        byte m_mode;
        data_type m_cropRatio;
    };
};

#endif
