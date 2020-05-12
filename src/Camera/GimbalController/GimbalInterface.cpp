#include "GimbalInterface.h"
#include "Joystick/JoystickLib/JoystickThreaded.h"
GimbalInterface::GimbalInterface(QObject *parent) : QObject(parent)
{
    m_context = new GimbalData();
}
JoystickThreaded* GimbalInterface::joystick(){
    return m_joystick;
}
void GimbalInterface::setJoystick(JoystickThreaded* joystick){
    m_joystick = joystick;
}
void GimbalInterface::connectToGimbal(Config* config){
    if(config != nullptr){
        m_config = config;
    }
}
void GimbalInterface::setVideoEngine(VideoEngine* videoEngine){
    m_videoEngine = videoEngine;
}
void GimbalInterface::disconnectGimbal(){

}
void GimbalInterface::discoverOnLan(){
}
void GimbalInterface::changeSensor(QString sensorID){

}
void GimbalInterface::handleAxes(){

}
void GimbalInterface::lockScreenPoint(int _id,
                double _px,double _py,double _oW,double _oH,
                double _w,double _h){

    float px = (_px + _oW/2) - _w/2;
    float py = (_py + _oH/2) - _h/2;
//    printf("%s [%.0f, %.0f]\r\n",__func__,px,py);
    float hfov = m_context->m_hfov[m_context->m_sensorID]/180.0*M_PI;

    float focalLength = _w / 2 / tan(hfov/2);

    float deltaPan = atan(-px/focalLength) * 180.0 / M_PI;
//            if(deltaPan > 10)deltaPan = 10
//            else if(deltaPan < -10)deltaPan = -10
    m_iPan+=deltaPan/30.0;
    m_cPan+=(m_panRate - m_uPan)/30.0;
    float dPan = (deltaPan - m_dPanOld) * 30;
    m_uPan = m_kpPan * deltaPan + m_kiPan * m_iPan + m_kdPan * dPan;
    m_dPanOld = deltaPan;

    m_panRate = m_uPan;

    float deltaTilt = atan(-py/focalLength) * 180.0 / M_PI;
//            if(deltaTilt > 10)deltaTilt = 10
//            else if(deltaTilt < -10)deltaTilt = -10
    m_iTilt+=deltaTilt/30.0;
    m_cTilt+=(m_tiltRate - m_uTilt)/30.0;
    float dTilt = (deltaTilt - m_dTiltOld) * 30;
    m_uTilt = m_kpTilt * deltaTilt + m_kiTilt * m_iTilt + m_kdTilt * dTilt;
    m_dTiltOld = deltaTilt;

    m_tiltRate = m_uTilt;
    setGimbalRate(m_panRate, m_tiltRate);
}
void GimbalInterface::setPanRate(float rate){
    Q_UNUSED(rate);
}
void GimbalInterface::setTiltRate(float rate){
    Q_UNUSED(rate);
}
void GimbalInterface::setGimbalRate(float panRate,float tiltRate){
    Q_UNUSED(panRate);
    Q_UNUSED(tiltRate);
}
void GimbalInterface::setPanPos(float pos){
    Q_UNUSED(pos);
}
void GimbalInterface::setTiltPos(float pos){
    Q_UNUSED(pos);
}
void GimbalInterface::setGimbalPos(float panPos,float tiltPos){
    Q_UNUSED(panPos);
    Q_UNUSED(tiltPos);
}
void GimbalInterface::setEOZoom(QString command, float value){
    Q_UNUSED(command);
    Q_UNUSED(value);
}
void GimbalInterface::setIRZoom(QString command){
    Q_UNUSED(command);
}
void GimbalInterface::snapShot(){

}
void GimbalInterface::changeTrackSize(float trackSize){
    Q_UNUSED(trackSize);
}
void GimbalInterface::setDigitalStab(bool enable){
    Q_UNUSED(enable);
}
void GimbalInterface::setRecord(bool enable){
    Q_UNUSED(enable);
}
void GimbalInterface::setShare(bool enable){
    Q_UNUSED(enable);
}
void GimbalInterface::setGimbalMode(QString mode){
    Q_UNUSED(mode);
}
void GimbalInterface::setGimbalPreset(QString mode){
    Q_UNUSED(mode);
}
void GimbalInterface::setGimbalRecorder(bool enable){
    Q_UNUSED(enable);
}
void GimbalInterface::setGCSRecorder(bool enable){
    Q_UNUSED(enable);
}
void GimbalInterface::setLockMode(QString mode, QPoint location){
    Q_UNUSED(mode);
    Q_UNUSED(location);
}
void GimbalInterface::setGeoLockPosition(QPoint location){
    Q_UNUSED(location);
}
