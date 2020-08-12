#include "VideoConverter.h"

VideoConverter::VideoConverter(QObject *parent) : QObject(parent)
{

}
VideoConverter::~VideoConverter()
{

}
void VideoConverter::start(){
    m_stop = false;
}
void VideoConverter::stop(){
    m_stop = true;
}
void VideoConverter::setState(QString state){

}
void VideoConverter::doWork(){

}
