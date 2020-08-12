#ifndef INSTALL_MODE_H
#define INSTALL_MODE_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"
#include "EyeStatus.h"

#define AP_GCS 3 // Virtual lock arround point screen
#define AP_AP 4 //

namespace Eye
{
    class InstallMode :public Object
    {
    private:
        byte m_mount;
        byte m_ap;
    public:
        InstallMode()
        {
            m_mount = (byte)Status::InstallMode::MOUNT_BELL;
            m_ap = AP_AP;
        }

        InstallMode(byte _mount, byte _ap)
        {
            m_mount = _mount;
            m_ap = _ap;
        }
        InstallMode(const InstallMode &_eye_mode)
        {
            m_mount = _eye_mode.m_mount;
            m_ap = _eye_mode.m_ap;
        }
        ~InstallMode(){}

        inline byte getMount()const
        {
            return m_mount;
        }

        inline void setMount(const byte _mount)
        {
            m_mount = _mount;
        }

        inline byte getAP() const
        {
            return m_ap;
        }

        inline void setAP(const byte &_mode)
        {
            m_ap = _mode;
        }

        inline InstallMode& operator =(const InstallMode& _eye_mode)
        {
            m_mount = _eye_mode.m_mount;
            m_ap = _eye_mode.m_ap;
            return *this;
        }

        inline bool operator ==(const InstallMode& _eye_mode)
        {
            return (m_mount == _eye_mode.m_mount&&
                m_ap == _eye_mode.m_ap);
        }

        length_type size()
        {
            return 2 * sizeof(byte);
        }

        std::vector<byte> toByte()
        {
            std::vector<byte> _result;
            _result.clear();
            _result.push_back(m_mount);
            _result.push_back(m_ap);
            return _result;
        }
        InstallMode* parse(std::vector<byte> _data, index_type _index = 0)
        {
            m_mount = _data[_index];
            m_ap = _data[_index + 1];
            return this;
        }

        InstallMode* parse(unsigned char* _data, index_type _index = 0)
        {
            m_mount = _data[_index];
            m_ap = _data[_index + 1];
            return this;
        }
    };
}

#endif
