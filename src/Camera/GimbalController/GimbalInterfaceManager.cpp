#include "GimbalInterfaceManager.h"

GimbalInterfaceManager::GimbalInterfaceManager(QObject *parent) : QObject(parent)
{

}
GimbalInterface* GimbalInterfaceManager::getGimbal(GIMBAL_TYPE type){
    if(type == GIMBAL_TYPE::CM160){
        return new CM160Gimbal;
    }else if(type == GIMBAL_TYPE::GREMSEY){
        return new GremseyGimbal;
    }else if(type == GIMBAL_TYPE::TRERON){
        return new TreronGimbal;
    }else
        return new GimbalInterface;
}
