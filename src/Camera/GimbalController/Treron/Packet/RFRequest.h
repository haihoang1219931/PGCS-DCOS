#ifndef RFREQUEST_H
#define RFREQUEST_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

#define RF_NONE 0
#define RF_LRF 1
#define RF_GPS 2
#define RF_UAV 3

namespace Eye
{
	class RFRequest :public Object
	{
	public:
        RFRequest() :m_mode(RF_LRF){}
		RFRequest(const index_type _frameId, const byte _mode)
		{
            m_frameId = _frameId;
            m_mode = _mode;
		}
        ~RFRequest(){}
    public:
        inline byte getMode() const{
            return m_mode;
        }
        inline void setMode(const byte _mode){
            m_mode = _mode;
        }

        inline index_type getFrameID() const{
            return m_frameId;
        }
        inline void setFrameID(const index_type _frameId){
            m_frameId = _frameId;
        }

        inline RFRequest & operator=(const RFRequest &_mode)
		{
			m_mode = _mode.m_mode;
			m_frameId = _mode.m_frameId;
			return *this;
		}

		inline bool operator==(const RFRequest &_mode)
		{
			return (m_mode == _mode.m_mode && m_frameId == _mode.m_frameId);
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
		RFRequest* parse(std::vector<byte> _data, index_type _index = 0)
        {
			byte* data = _data.data();
			m_mode = data[_index];
			m_frameId = Utils::toValue<index_type>(data, _index + 1);
			return this;
		}
		RFRequest* parse(byte* _data, index_type _index = 0)
        {
			m_mode = _data[_index];
			m_frameId = Utils::toValue<index_type>(_data, _index + 1);
			return this;
		}
	private:
		index_type m_frameId;
		byte m_mode;
	};
}
#endif
