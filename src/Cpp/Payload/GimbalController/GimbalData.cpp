#include "GimbalData.h"

GimbalData::GimbalData(QObject *parent) : QObject(parent)
{
    m_hfovMax[0] = 63.1f;
    m_hfovMax[1] = 17.7f;
    m_hfovMin[0] = 2.33f;
    m_hfovMin[1] = 17.7f;
    m_hfov[0] = 63.1f;
    m_hfov[1] = 17.7f;
    m_zoom[0] = 1;
    setCorners(0,0,0,0,0,0,0,0,0,0);
    m_mapLockMode["FREE"] = 0;
    m_mapLockMode["TRACK"] = 1;
    m_mapLockMode["VISUAL"] = 2;
}
