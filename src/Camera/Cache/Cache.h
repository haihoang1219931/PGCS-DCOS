#ifndef CACHE_H
#define CACHE_H

#include "../ControllerLib/Buffer/RollBuffer.h"
#include "../ControllerLib/Buffer/RollBuffer_.h"
#include "../ControllerLib/Packet/MotionImage.h"
#include "../ControllerLib/Packet/SystemStatus.h"
#include "../ControllerLib/Packet/TrackResponse.h"
#include "../ControllerLib/Packet/XPoint.h"

#include "DetectedObjectsCacheItem.h"
#include "GstFrameCacheItem.h"
#include "ProcessImageCacheItem.h"
class Cache;
namespace rva
{
    class Cache
    {
        public:

            explicit Cache() {}

            static rva::Cache* instance()
            {
                if (nullptr == m_instance)
                {
                    m_instance = new rva::Cache();
                    m_instance->init();
                }

                return m_instance;
            }


            RollBuffer_<GstFrameCacheItem>* getGstFrameCache()
            {
                if (nullptr != m_instance)
                {
                    return m_gstFrameBuff;
                }

                return nullptr;
            }

            RollBuffer_<GstFrameCacheItem>* getGstRTSPCache()
            {
                if (nullptr != m_instance)
                {
                    return m_gstRTSPBuff;
                }

                return nullptr;
            }

            RollBuffer_<GstFrameCacheItem>* getGstEOSavingCache()
            {
                if (nullptr != m_instance)
                {
                    return m_gstEOSavingBuff;
                }

                return nullptr;
            }

            RollBuffer_<GstFrameCacheItem>* getGstIRSavingCache()
            {
                if (nullptr != m_instance)
                {
                    return m_gstIRSavingBuff;
                }

                return nullptr;
            }

            RollBuffer_<ProcessImageCacheItem>* getProcessImageCache()
            {
                if (nullptr != m_instance)
                {
                    return m_matImageBuff;
                }

                return nullptr;
            }

            RollBuffer<Eye::SystemStatus>* getSystemStatusCache()
            {
                if (nullptr != m_instance)
                {
                    return m_rbSystem;
                }
                return nullptr;
            }

            RollBuffer<Eye::MotionImage>* getMotionImageEOCache()
            {
                if (nullptr != m_instance)
                {
                    return m_rbIPCEO;
                }
                return nullptr;
            }

            RollBuffer<Eye::MotionImage>* getMotionImageIRCache()
            {
                if (nullptr != m_instance)
                {
                    return m_rbIPCIR;
                }
                return nullptr;
            }

            RollBuffer<Eye::TrackResponse>* getEOTrackingCache()
            {
                if (nullptr != m_instance)
                {
                    return m_rbTrackResEO;
                }
                return nullptr;
            }

            RollBuffer<Eye::XPoint>* getEOSteeringCache()
            {
                if (nullptr != m_instance)
                {
                    return m_rbXPointEO;
                }
                return nullptr;
            }

            RollBuffer<Eye::TrackResponse>* getIRTrackingCache()
            {
                if (nullptr != m_instance)
                {
                    return m_rbTrackResIR;
                }
                return nullptr;
            }

            RollBuffer<Eye::XPoint>* getIRSteeringCache()
            {
                if (nullptr != m_instance)
                {
                    return m_rbXPointIR;
                }
                return nullptr;
            }

            RollBuffer_<DetectedObjectsCacheItem>* getDetectedObjectsCache()
            {
                if (nullptr != m_instance)
                {
                    return m_rbDetectedObjs;
                }

                return nullptr;
            }

        private:
            void init()
            {
                m_gstFrameBuff = new RollBuffer_<GstFrameCacheItem>(30);
                m_gstRTSPBuff = new RollBuffer_<GstFrameCacheItem>(30);
                m_matImageBuff = new RollBuffer_<ProcessImageCacheItem>(30);
                m_gstIRSavingBuff = new RollBuffer_<GstFrameCacheItem>(10);
                m_gstEOSavingBuff = new RollBuffer_<GstFrameCacheItem>(10);
                m_rbSystem = new RollBuffer<Eye::SystemStatus>(60);
                m_rbIPCEO = new RollBuffer<Eye::MotionImage>(60);
                m_rbIPCIR = new RollBuffer<Eye::MotionImage>(60);
                m_rbTrackResEO = new RollBuffer<Eye::TrackResponse>(60);
                m_rbXPointEO = new RollBuffer<Eye::XPoint>(60);
                m_rbTrackResIR = new RollBuffer<Eye::TrackResponse>(60);
                m_rbXPointIR = new RollBuffer<Eye::XPoint>(60);
                m_rbDetectedObjs = new RollBuffer_<DetectedObjectsCacheItem>(60);
            }

        private:
            static rva::Cache* m_instance;
            RollBuffer_<GstFrameCacheItem>* m_gstFrameBuff;
            RollBuffer_<GstFrameCacheItem>* m_gstRTSPBuff;
            RollBuffer_<GstFrameCacheItem>* m_gstEOSavingBuff;
            RollBuffer_<GstFrameCacheItem>* m_gstIRSavingBuff;
            RollBuffer_<ProcessImageCacheItem>* m_matImageBuff;
            RollBuffer<Eye::SystemStatus>* m_rbSystem;
            RollBuffer<Eye::MotionImage>* m_rbIPCEO;
            RollBuffer<Eye::MotionImage>* m_rbIPCIR;
            RollBuffer<Eye::TrackResponse>* m_rbTrackResEO;
            RollBuffer<Eye::XPoint>* m_rbXPointEO;
            RollBuffer<Eye::TrackResponse>* m_rbTrackResIR;
            RollBuffer<Eye::XPoint>* m_rbXPointIR;
            RollBuffer_<DetectedObjectsCacheItem>* m_rbDetectedObjs;
    };
} // namespace rva
#endif // CACHE_H
