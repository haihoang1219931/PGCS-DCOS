#include "GimbalController.h"

GimbalController::GimbalController(QObject *parent) : QObject(parent)
{

}
GimbalInterface* GimbalController::gimbal(){
    return m_gimbal;
}
GimbalData* GimbalController::data(){
    return  m_data;
}
void GimbalController::connectToGimbal(Config* config){

}
void GimbalController::disConnectToGimbal(Config* config){

}
