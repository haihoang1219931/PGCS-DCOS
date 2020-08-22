#ifndef SNAP_SHOT_H
#define SNAP_SHOT_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"


namespace Eye
{
    class Snapshot :public Object
    {
    public:
        Snapshot(){}
        Snapshot(const index_type _frameId, const byte _mode)
        {
            m_frameId = _frameId;
            m_mode = _mode;
        }
        ~Snapshot(){}
    public:
        //
        inline index_type getFrameID()const
        {
            return m_frameId;
        }

        inline byte getMode()const
        {
            return m_mode;
        }

        inline void setMode(const byte _mode)
        {
            m_mode = _mode;
        }

        inline void setFrameID(const index_type _frameId)
        {
            m_frameId = _frameId;
        }

        inline Snapshot& operator =(const Snapshot& _mode)
        {
            m_mode = _mode.m_mode;
            m_frameId = _mode.m_frameId;
            return *this;
        }

        inline bool operator ==(const Snapshot& _mode)
        {
            return (m_mode == _mode.m_mode &&
                    m_frameId == _mode.m_frameId);
        }

        length_type size()
        {
            return sizeof(m_mode) + sizeof(m_frameId);
        }
        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            std::vector<byte> b_frameId;
            b_frameId = Utils::toByte<index_type>(m_frameId);
            _result.clear();
            _result.push_back(m_mode);
            _result.insert(_result.end(),b_frameId.begin(), b_frameId.end());
            return _result;
        }
        Snapshot* parse(std::vector<byte> _data, index_type _index = 0)
        {
            byte* data = _data.data();
            m_mode = Utils::toValue<byte>(data, _index);
            m_frameId = Utils::toValue<index_type>(data, _index + sizeof(m_mode));
            return this;
        }
        Snapshot* parse(byte* _data, index_type _index = 0)
        {
            m_mode = Utils::toValue<byte>(_data, _index);
            m_frameId = Utils::toValue<index_type>(_data, _index + sizeof(m_mode));
            return this;
        }
    private:
        byte m_mode;
        index_type m_frameId;
    };
}
#endif
