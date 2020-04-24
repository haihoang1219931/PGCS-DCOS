#include "GimbalInterface.h"

GimbalInterface::GimbalInterface(QObject *parent) : QObject(parent)
{
    m_context = new GimbalData();
}
void GimbalInterface::connectToGimbal(Config* config){
    if(config != nullptr){
        m_config = config;
    }
}
void GimbalInterface::disconnectGimbal(){

}
void GimbalInterface::discoverOnLan(){
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
void GimbalInterface::setEOZoom(QString command, QString value){
    Q_UNUSED(command);
    Q_UNUSED(value);
}
void GimbalInterface::setIRZoom(QString command){
    Q_UNUSED(command);
}
void GimbalInterface::snapShot(){

}
void GimbalInterface::setGimbalMode(QString mode){
    Q_UNUSED(mode);
}
void GimbalInterface::setGimbalPreset(QString mode){
    Q_UNUSED(mode);
}
void GimbalInterface::setGimbalRecorder(QString mode){
    Q_UNUSED(mode);
}
void GimbalInterface::setLockMode(QString mode, bool enable, QPoint location){
    Q_UNUSED(mode);
    Q_UNUSED(enable);
    Q_UNUSED(location);
}
void GimbalInterface::setGeoLockPosition(QPoint location){
    Q_UNUSED(location);
}
