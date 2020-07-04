#ifndef SCENESTEERING_H
#define SCENESTEERING_H

#include "Object.h"
#include "utils.h"
#include "Common_type.h"

namespace Eye {
class SceneSteering :public Object
{
public:
    SceneSteering(){
        m_id = -1;
    }

    SceneSteering(const index_type _id)
    {
        m_id = _id;
    }

    ~SceneSteering(){}

public:
    //
    inline index_type getID()const
    {
        return m_id;
    }

    inline void setID(const index_type _id)
    {
        m_id = _id;
    }

    inline SceneSteering& operator =(const SceneSteering& _mode)
    {
        m_id = _mode.m_id;
        return *this;
    }

    inline bool operator ==(const SceneSteering& _mode)
    {
        return m_id == _mode.m_id;
    }

    length_type size()
    {
        return sizeof(m_id);
    }
    std::vector<byte> toByte()
    {
        std::vector<byte> _result = Utils::toByte<index_type>(m_id);
        return _result;
    }
    SceneSteering* parse(std::vector<byte> _data, index_type _index = 0)
    {
        byte* data = _data.data();
        m_id = Utils::toValue<index_type>(data, _index);
        return this;
    }
    SceneSteering* parse(byte* _data, index_type _index = 0)
    {
        m_id = Utils::toValue<index_type>(_data, _index);
        return this;
    }
private:
    index_type m_id;
};
}


#endif // SCENESTEERING_H
