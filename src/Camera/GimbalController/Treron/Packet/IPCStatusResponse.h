#ifndef IPCRESPONSE_H
#define IPCRESPONSE_H

#include "Common_type.h"
#include "Object.h"
#include "utils.h"
#include "EyeStatus.h"

namespace Eye
{
    class IPCStatusResponse : public Object
    {
        public:
            IPCStatusResponse() {}
            IPCStatusResponse(const data_type _freeMemory, const data_type _totalMemory,
                              const data_type _hfovEO, const data_type _hfovEOIR,
                              const byte _sensorColorMode, const byte _sensorMode,
                              const byte _videoStabEn, const byte _snapShotMode,
                              const data_type _imageResizeWidth, const data_type _imageResizeHeight,
                              const data_type _resolutionWidth, const data_type _resolutionHeight,
                              const data_type _cropMode, const byte _lockMode, const byte _recordMode,
                              const data_type _trackSize)
                : m_freeMemory(_freeMemory), m_totalMemory(_totalMemory),
                  m_hfovEO(_hfovEO), m_hfovEOIR(_hfovEOIR), m_sensorColorMode(_sensorColorMode),
                  m_sensorMode(_sensorMode), m_videoStabEn(_videoStabEn),
                  m_snapShotMode(_snapShotMode), m_imageResizeWidth(_imageResizeWidth),
                  m_imageResizeHeight(_imageResizeHeight), m_resolutionWidth(_resolutionWidth),
                  m_resolutionHeight(_resolutionHeight), m_cropMode(_cropMode),
                  m_lockMode(_lockMode), m_recordMode(_recordMode), m_trackSize(_trackSize)
            {
            }

            ~IPCStatusResponse() {}
            void setLockMode(byte lockMode){
                m_lockMode = lockMode;
            }
        public:
            data_type m_freeMemory;
            data_type m_totalMemory;
            data_type m_hfovEO;
            data_type m_hfovEOIR;
            byte m_sensorColorMode;
            byte m_sensorMode;
            byte m_videoStabEn;
            byte m_snapShotMode;
            data_type m_imageResizeWidth;
            data_type m_imageResizeHeight;
            data_type m_resolutionWidth;
            data_type m_resolutionHeight;
            data_type m_cropMode;
            byte m_lockMode;
            byte m_recordMode;
            data_type m_trackSize;

        public:
            data_type getFreeMemory()
            {
                return m_freeMemory;
            }

            data_type getTotalMemory()
            {
                return m_totalMemory;
            }

            data_type getHFOV()
            {
                return m_hfovEO;
            }

            data_type getVFOV()
            {
                return m_hfovEOIR;
            }

            byte getSensorColorMode()
            {
                return m_sensorColorMode;
            }

            byte getSensorMode()
            {
                return m_sensorMode;
            }

            byte getVideoStabEn()
            {
                return m_videoStabEn;
            }

            byte getSnapShotMode()
            {
                return m_snapShotMode;
            }

            data_type getImageResizeWidth()
            {
                return m_imageResizeWidth;
            }

            data_type getImageResizeHeight()
            {
                return m_imageResizeHeight;
            }

            data_type getResolutionWidth()
            {
                return m_resolutionWidth;
            }

            data_type getResolutionHeight()
            {
                return m_resolutionHeight;
            }

            data_type getCropMode()
            {
                return m_cropMode;
            }

            byte getLockMode()
            {
                return m_lockMode;
            }

            byte getRecordMode()
            {
                return m_recordMode;
            }

            data_type getTrackSize()
            {
                return m_trackSize;
            }

            length_type size()
            {
                return 9 * sizeof(data_type) + 6 * sizeof(byte);
            }

            std::vector<byte> toByte()
            {
                std::vector<byte> result;
                std::vector<byte> b_temp;
                result = Utils::toByte<data_type>(m_freeMemory);
                b_temp = Utils::toByte<data_type>(m_totalMemory);
                result.insert(result.end(), b_temp.begin(), b_temp.end());
                b_temp = Utils::toByte<data_type>(m_hfovEO);
                result.insert(result.end(), b_temp.begin(), b_temp.end());
                b_temp = Utils::toByte<data_type>(m_hfovEOIR);
                result.insert(result.end(), b_temp.begin(), b_temp.end());
                result.push_back(m_sensorColorMode);
                result.push_back(m_sensorMode);
                result.push_back(m_videoStabEn);
                result.push_back(m_snapShotMode);
                b_temp = Utils::toByte<data_type>(m_imageResizeWidth);
                result.insert(result.end(), b_temp.begin(), b_temp.end());
                b_temp = Utils::toByte<data_type>(m_imageResizeHeight);
                result.insert(result.end(), b_temp.begin(), b_temp.end());
                b_temp = Utils::toByte<data_type>(m_resolutionWidth);
                result.insert(result.end(), b_temp.begin(), b_temp.end());
                b_temp = Utils::toByte<data_type>(m_resolutionHeight);
                result.insert(result.end(), b_temp.begin(), b_temp.end());
                b_temp = Utils::toByte<data_type>(m_cropMode);
                result.insert(result.end(), b_temp.begin(), b_temp.end());
                result.push_back(m_lockMode);
                result.push_back(m_recordMode);
                b_temp = Utils::toByte<data_type>(m_trackSize);
                result.insert(result.end(), b_temp.begin(), b_temp.end());
                return result;
            }

            IPCStatusResponse* parse(byte* _data, index_type _index = 0)
            {
                m_freeMemory = Utils::toValue<data_type>(_data, _index);
                m_totalMemory = Utils::toValue<data_type>(_data, _index + sizeof(data_type));
                m_hfovEO = Utils::toValue<data_type>(_data, _index + 2 * sizeof(data_type));
                m_hfovEOIR = Utils::toValue<data_type>(_data, _index + 3 * sizeof(data_type));
                m_sensorColorMode = Utils::toValue<byte>(_data, _index + 4 * sizeof(data_type));
                m_sensorMode = Utils::toValue<byte>(_data, _index + sizeof(byte) + 4 * sizeof(data_type));
                m_videoStabEn = Utils::toValue<byte>(_data, _index + 2 * sizeof(byte) + 4 * sizeof(data_type));
                m_snapShotMode = Utils::toValue<byte>(_data, _index + 3 * sizeof(byte) + 4 * sizeof(data_type));
                m_imageResizeWidth = Utils::toValue<data_type>(_data, _index + 4 * sizeof(byte) + 4 * sizeof(data_type));
                m_imageResizeHeight = Utils::toValue<data_type>(_data, _index + 4 * sizeof(byte) + 5 * sizeof(data_type));
                m_resolutionWidth = Utils::toValue<data_type>(_data, _index + 4 * sizeof(byte) + 6 * sizeof(data_type));
                m_resolutionHeight = Utils::toValue<data_type>(_data, _index + 4 * sizeof(byte) + 7 * sizeof(data_type));
                m_cropMode = Utils::toValue<data_type>(_data, _index + 4 * sizeof(byte) + 8 * sizeof(data_type));
                m_lockMode = Utils::toValue<byte>(_data, _index + 4 * sizeof(byte) + 9 * sizeof(data_type));
                m_recordMode = Utils::toValue<byte>(_data, _index + 5 * sizeof(byte) + 9 * sizeof(data_type));
                m_trackSize = Utils::toValue<data_type>(_data, _index + 6 * sizeof(byte) + 9 * sizeof(data_type));
                return this;
            }

            IPCStatusResponse* parse(std::vector<byte> data, index_type _index = 0)
            {
                byte* _data = data.data();
                m_freeMemory = Utils::toValue<data_type>(_data, _index);
                m_totalMemory = Utils::toValue<data_type>(_data, _index + sizeof(data_type));
                m_hfovEO = Utils::toValue<data_type>(_data, _index + 2 * sizeof(data_type));
                m_hfovEOIR = Utils::toValue<data_type>(_data, _index + 3 * sizeof(data_type));
                m_sensorColorMode = Utils::toValue<byte>(_data, _index + 4 * sizeof(data_type));
                m_sensorMode = Utils::toValue<byte>(_data, _index + sizeof(byte) + 4 * sizeof(data_type));
                m_videoStabEn = Utils::toValue<byte>(_data, _index + 2 * sizeof(byte) + 4 * sizeof(data_type));
                m_snapShotMode = Utils::toValue<byte>(_data, _index + 3 * sizeof(byte) + 4 * sizeof(data_type));
                m_imageResizeWidth = Utils::toValue<data_type>(_data, _index + 4 * sizeof(byte) + 4 * sizeof(data_type));
                m_imageResizeHeight = Utils::toValue<data_type>(_data, _index + 4 * sizeof(byte) + 5 * sizeof(data_type));
                m_resolutionWidth = Utils::toValue<data_type>(_data, _index + 4 * sizeof(byte) + 6 * sizeof(data_type));
                m_resolutionHeight = Utils::toValue<data_type>(_data, _index + 4 * sizeof(byte) + 7 * sizeof(data_type));
                m_cropMode = Utils::toValue<data_type>(_data, _index + 4 * sizeof(byte) + 8 * sizeof(data_type));
                m_lockMode = Utils::toValue<byte>(_data, _index + 4 * sizeof(byte) + 9 * sizeof(data_type));
                m_recordMode = Utils::toValue<byte>(_data, _index + 5 * sizeof(byte) + 9 * sizeof(data_type));
                m_trackSize = Utils::toValue<data_type>(_data, _index + 6 * sizeof(byte) + 9 * sizeof(data_type));
                return this;
            }
    };
}

#endif // IPCRESPONSE_H
