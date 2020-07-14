#ifndef GIMBAL_RECORD_STATUS_H
#define GIMBAL_RECORD_STATUS_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"

#define RECORD_FULL 1 //all sensor on IPC
#define RECORD_OFF 2
#define RECORD_VIEW 3
#define RECORD_TYPE_A 4
#define RECORD_TYPE_B 5

namespace Eye
{
	class GimbalRecordStatus :public Object
	{
	public:
        GimbalRecordStatus(){}
		GimbalRecordStatus(const byte _mode, const byte _type, const data_type &_totalMemory, const data_type &_freeMemory)
		{
		}
        ~GimbalRecordStatus(){}
	private:
		byte m_mode;
		byte m_type;
		data_type m_totalMemory;
		data_type m_freeMemory;
	};
}
#endif
