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

void GimbalInterface::setVehicle(Vehicle *vehicle)
{
    Q_UNUSED(vehicle);
}
void GimbalInterface::enableSensor(QString sensorID, bool enable){
    Q_UNUSED(sensorID);
    Q_UNUSED(enable);
}
void GimbalInterface::disconnectGimbal(){

}
void GimbalInterface::discoverOnLan(){
}
void GimbalInterface::changeSensor(QString sensorID){

}
void GimbalInterface::setSensorColor(QString sensorID,QString colorMode){

}
void GimbalInterface::handleAxes(){

}

void GimbalInterface::lockScreenPoint(int _id,
                                      double _px,double _py,double _oW,double _oH,
                                      double _w,double _h){
//    double deadRate = 8;
//    if(fabs(m_panRate) > deadRate || fabs(m_tiltRate) > deadRate)
    {
        m_beginTime = clock();

        double T = static_cast<double>(m_beginTime - m_endTime)/CLOCKS_PER_SEC;
        m_endTime = m_beginTime ;

        double fps = 1/T;

        double px = (_px + _oW/2) - _w/2;
        double py = (_py + _oH/2) - _h/2;
        double hfov = static_cast<double>(m_context->m_hfov[m_context->m_sensorID])/180.0*M_PI;
//        printf("%s [%.0f, %.0f] hfov=%f fps=%f\r\n",__func__,px,py,hfov,fps);
        double focalLength = _w / 2 / tan(hfov/2);

        double deltaPan = atan(-px/focalLength) * 180.0 / M_PI;

        printf("delta pan: %.4f\r\n",deltaPan);
        //            if(deltaPan > 10)deltaPan = 10
        //            else if(deltaPan < -10)deltaPan = -10
        double min_focal_length = _w / 2 / tan(70.2/2);
        double m_pan_limit = (min_focal_length / focalLength) *m_pan_limit_i_default;
        printf("limit pan: %.4f\r\n",m_pan_limit);
        if(deltaPan<m_pan_limit && deltaPan>-m_pan_limit)
            m_iPan+=deltaPan/fps;
        else
            m_iPan = 0;
        m_cPan+=(m_panRate - m_uPan)/fps;
        double dPan = (deltaPan - m_dPanOld) * fps;
        m_uPan = m_kpPan * deltaPan + m_kiPan * m_iPan + m_kdPan * dPan;
        m_dPanOld = deltaPan;

        m_panRate = m_uPan;

//        printf("I pan: %.4f\r\n",m_kiPan * m_iPan);

        double deltaTilt = atan(-py/focalLength) * 180.0 / M_PI;
        //            if(deltaTilt > 10)deltaTilt = 10
        //            else if(deltaTilt < -10)deltaTilt = -10
        if(deltaTilt<0.1 && deltaTilt>-0.1)
            m_iTilt+=deltaTilt/fps;
        else
            m_iTilt = 0;
        m_cTilt+=(m_tiltRate - m_uTilt)/fps;
        double dTilt = (deltaTilt - m_dTiltOld) * fps;
        m_uTilt = m_kpTilt * deltaTilt + m_kiTilt * m_iTilt + m_kdTilt * dTilt;
        m_dTiltOld = deltaTilt;

        m_tiltRate =  m_uTilt;

//        if(m_panRate<8 && m_panRate >1)
//        {
//            m_panRate=8;
//        }
//        else if(m_panRate> -8 && m_panRate <-1)
//        {
//               m_panRate = -8;
//        }

//        if(m_tiltRate<8 && m_tiltRate >1)
//        {
//            m_tiltRate=8;
//        }
//        else if(m_tiltRate> -8 && m_tiltRate <-1)
//        {
//               m_tiltRate = -8;
//        }

//        printf("%s _px=%f _py=%f _oW=%f _oH=%f _w=%f _h=%f => panRate=%d tiltRate=%d\r\n",
//               __func__,_px,_py,_oW,_oH,_w,_h,
//               static_cast<int>(m_panRate),
//               static_cast<int>(m_tiltRate));

        setGimbalRate(static_cast<float>(m_panRate),
                      static_cast<float>(m_tiltRate));
    }

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
void GimbalInterface::setDefog(QString mode){
    Q_UNUSED(mode);
}
void GimbalInterface::setGimbalRecorder(bool enable){
    Q_UNUSED(enable);
}
void GimbalInterface::setGCSRecorder(bool enable){
    Q_UNUSED(enable);
}
void GimbalInterface::setLockMode(QString mode, QPointF location){
    Q_UNUSED(mode);
    Q_UNUSED(location);
}
void GimbalInterface::setGeoLockPosition(QPoint location){
    Q_UNUSED(location);
}
void GimbalInterface::setObjectSearch(bool enable){
    Q_UNUSED(enable);
}
void GimbalInterface::handleGimbalModeChanged(QString mode)
{
    Q_UNUSED(mode);
}

void GimbalInterface::handleGimbalSetModeFail()
{

}
