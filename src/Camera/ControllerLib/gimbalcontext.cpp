#include "gimbalcontext.h"

GimbalContext::GimbalContext(QObject* parent) : QObject(parent)
{
    m_rbSystem = rva::Cache::instance()->getSystemStatusCache();
    m_rbIPCEO = rva::Cache::instance()->getMotionImageEOCache();
    m_rbIPCIR = rva::Cache::instance()->getMotionImageIRCache();
    m_rbTrackResEO = rva::Cache::instance()->getEOTrackingCache();
    m_rbXPointEO = rva::Cache::instance()->getEOSteeringCache();
    m_rbTrackResIR = rva::Cache::instance()->getIRTrackingCache();
    m_rbXPointIR = rva::Cache::instance()->getIRSteeringCache();
    m_zoomMax[0] = 60.08f;
    m_zoomMax[1] = 17.7f;
    m_zoomMin[0] = 2.08f;
    m_zoomMin[1] = 0.4f;
    m_hfov[1] = 0.626888;
}
GimbalContext::~GimbalContext() {}
