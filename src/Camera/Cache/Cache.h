#ifndef CACHE_H
#define CACHE_H

#include "Camera/Buffer/RollBuffer.h"

#include "DetectedObjectsCacheItem.h"
#include "GstFrameCacheItem.h"
#include "ProcessImageCacheItem.h"
namespace rva
{
class Cache
{
    public:
        explicit Cache() {}

        static Cache *instance()
        {
            if (nullptr == m_instance) {
                m_instance = new Cache;
                m_instance->init();
            }

            return m_instance;
        }

        RollBuffer<GstFrameCacheItem> *getGstFrameCache()
        {
            if (nullptr != m_instance) {
                return m_gstFrameBuff;
            }

            return nullptr;
        }

        RollBuffer<ProcessImageCacheItem> *getProcessImageCache()
        {
            if (nullptr != m_instance) {
                return m_matImageBuff;
            }

            return nullptr;
        }

        RollBuffer<ProcessImageCacheItem> *getTrackImageCache()
        {
            if (nullptr != m_instance) {
                return m_matTrackBuff;
            }

            return nullptr;
        }

//        RollBuffer<Eye::SystemStatus> *getSystemStatusCache()
//        {
//            if (nullptr != m_instance) {
//                return m_rbSystem;
//            }
//            return nullptr;
//        }

//        RollBuffer<Eye::MotionImage> *getMotionImageEOCache()
//        {
//            if (nullptr != m_instance) {
//                return m_rbIPCEO;
//            }
//            return nullptr;
//        }

//        RollBuffer<Eye::MotionImage> *getMotionImageIRCache()
//        {
//            if (nullptr != m_instance) {
//                return m_rbIPCIR;
//            }
//            return nullptr;
//        }

//        RollBuffer<Eye::TrackResponse> *getEOTrackingCache()
//        {
//            if (nullptr != m_instance) {
//                return m_rbTrackResEO;
//            }
//            return nullptr;
//        }

//        RollBuffer<Eye::XPoint> *getEOSteeringCache()
//        {
//            if (nullptr != m_instance) {
//                return m_rbXPointEO;
//            }
//            return nullptr;
//        }

//        RollBuffer<Eye::TrackResponse> *getIRTrackingCache()
//        {
//            if (nullptr != m_instance) {
//                return m_rbTrackResIR;
//            }
//            return nullptr;
//        }

//        RollBuffer<Eye::XPoint> *getIRSteeringCache()
//        {
//            if (nullptr != m_instance) {
//                return m_rbXPointIR;
//            }
//            return nullptr;
//        }

        RollBuffer<DetectedObjectsCacheItem> *getDetectedObjectsCache()
        {
            if (nullptr != m_instance) {
                return m_rbDetectedObjs;
            }

            return nullptr;
        }

        RollBuffer<DetectedObjectsCacheItem> *getMOTCache()
        {
            if (nullptr != m_instance) {
                return m_rbMOTObjs;
            }

            return nullptr;
        }

        RollBuffer<DetectedObjectsCacheItem> *getSearchCache()
        {
            if (nullptr != m_instance) {
                return m_rbSearchObjs;
            }

            return nullptr;
        }

        RollBuffer<GstFrameCacheItem> *getGstRTSPCache()
        {
            if (nullptr != m_instance) {
                return m_gstRTSPBuff;
            }

            return nullptr;
        }

        RollBuffer<GstFrameCacheItem> *getGstEOSavingCache()
        {
            if (nullptr != m_instance) {
                return m_gstEOSavingBuff;
            }

            return nullptr;
        }

        RollBuffer<GstFrameCacheItem> *getGstIRSavingCache()
        {
            if (nullptr != m_instance) {
                return m_gstIRSavingBuff;
            }

            return nullptr;
        }
    private:
        void init()
        {
            m_gstFrameBuff = new RollBuffer<GstFrameCacheItem>(30);
            m_gstRTSPBuff = new RollBuffer<GstFrameCacheItem>(30);
            m_gstIRSavingBuff = new RollBuffer<GstFrameCacheItem>(10);
            m_gstEOSavingBuff = new RollBuffer<GstFrameCacheItem>(10);
            m_matImageBuff = new RollBuffer<ProcessImageCacheItem>(30);
            m_matTrackBuff = new RollBuffer<ProcessImageCacheItem>(20);
//            m_rbSystem = new RollBuffer<Eye::SystemStatus>(60);
//            m_rbIPCEO = new RollBuffer<Eye::MotionImage>(60);
//            m_rbIPCIR = new RollBuffer<Eye::MotionImage>(60);
//            m_rbTrackResEO = new RollBuffer<Eye::TrackResponse>(60);
//            m_rbXPointEO = new RollBuffer<Eye::XPoint>(60);
//            m_rbTrackResIR = new RollBuffer<Eye::TrackResponse>(60);
//            m_rbXPointIR = new RollBuffer<Eye::XPoint>(60);
            m_rbDetectedObjs = new RollBuffer<DetectedObjectsCacheItem>(60);
            m_rbMOTObjs = new RollBuffer<DetectedObjectsCacheItem>(60);
            m_rbSearchObjs = new RollBuffer<DetectedObjectsCacheItem>(60);
        }

    private:
        static Cache *m_instance;
        RollBuffer<GstFrameCacheItem> *m_gstFrameBuff;
        RollBuffer<GstFrameCacheItem> *m_gstRTSPBuff;
        RollBuffer<GstFrameCacheItem> *m_gstEOSavingBuff;
        RollBuffer<GstFrameCacheItem> *m_gstIRSavingBuff;
        RollBuffer<ProcessImageCacheItem> *m_matImageBuff;
        RollBuffer<ProcessImageCacheItem> *m_matTrackBuff;
//        RollBuffer<Eye::SystemStatus> *m_rbSystem;
//        RollBuffer<Eye::MotionImage> *m_rbIPCEO;
//        RollBuffer<Eye::MotionImage> *m_rbIPCIR;
//        RollBuffer<Eye::TrackResponse> *m_rbTrackResEO;
//        RollBuffer<Eye::XPoint> *m_rbXPointEO;
//        RollBuffer<Eye::TrackResponse> *m_rbTrackResIR;
//        RollBuffer<Eye::XPoint> *m_rbXPointIR;
        RollBuffer<DetectedObjectsCacheItem> *m_rbDetectedObjs;
        RollBuffer<DetectedObjectsCacheItem> *m_rbMOTObjs;
        RollBuffer<DetectedObjectsCacheItem> *m_rbSearchObjs;
};
} // namespace rva
#endif // CACHE_H
