#include "VideoEngineInterface.h"
#include "../VideoDisplay/VideoRender.h"
#include "VRTSPServer.h"
#include "VSavingWorker.h"
#include "../Cache/TrackObject.h"
#include "Joystick/JoystickLib/JoystickThreaded.h"
#include "../GimbalController/GimbalInterface.h"
VideoEngine::VideoEngine(QObject *parent) : QObject(parent)
{

}
GimbalInterface* VideoEngine::gimbal(){
    return m_gimbal;
}
void VideoEngine::setGimbal(GimbalInterface* gimbal){
    m_gimbal = gimbal;
    m_gimbal->setZoomTarget(0,m_gimbal->context()->m_zoom[0]);
    m_gimbal->setZoomCalculated(0,m_gimbal->context()->m_zoom[0]);
}
void VideoEngine::loadConfig(Config* config){
    if(config != nullptr){
        m_config = config;
        setVideo(m_config->value("Settings:StreamEO:Value:data").toString());
        start();
    }
}
QQmlListProperty<TrackObjectInfo> VideoEngine::listTrackObjectInfos()
{
    return QQmlListProperty<TrackObjectInfo>(this, m_listTrackObjectInfos);
}
void VideoEngine::addTrackObjectInfo(TrackObjectInfo* object)
{
    this->m_listTrackObjectInfos.append(object);
    Q_EMIT listTrackObjectInfosChanged();
}
void VideoEngine::removeTrackObjectInfo(const int& sequence) {
    if(sequence < 0 || sequence >= this->m_listTrackObjectInfos.size()){
        return;
    }

    // remove user on list
    this->m_listTrackObjectInfos.removeAt(sequence);
    Q_EMIT listTrackObjectInfosChanged();
}
void VideoEngine::removeTrackObjectInfo(const QString &userUid)
{
    // check room contain user
    int sequence = -1;
    for (int i = 0; i < this->m_listTrackObjectInfos.size(); i++) {
        if (this->m_listTrackObjectInfos[i]->userId() == userUid) {
            sequence = i;
            break;
        }
    }
    removeTrackObjectInfo(sequence);
}
void VideoEngine::updateTrackObjectInfo(const QString& userUid, const QString& attr, const QVariant& newValue) {

    for(int i = 0; i < this->m_listTrackObjectInfos.size(); i++ ) {
        TrackObjectInfo* object = this->m_listTrackObjectInfos[i];
        if(userUid.contains(this->m_listTrackObjectInfos.at(i)->userId())) {
            if( attr == "RECT"){
                object->setRect(newValue.toRect());
            }else if( attr == "SIZE"){
                object->setSourceSize(newValue.toSize());
            }else if( attr == "LATITUDE"){
                object->setLatitude(newValue.toFloat());
            }else if( attr == "LONGTITUDE"){
                object->setLongitude(newValue.toFloat());
            }else if( attr == "SPEED"){
                object->setSpeed(newValue.toFloat());
            }else if( attr == "ANGLE"){
                object->setAngle(newValue.toFloat());
            }else if( attr == "SCREEN_X"){
                object->setScreenX(newValue.toInt());
            }else if( attr == "SCREEN_Y"){
                object->setScreenY(newValue.toInt());
            }
            if( attr == "SELECTED"){
                object->setIsSelected(newValue.toBool());
            }
        }else{
            if( attr == "SELECTED"){
                object->setIsSelected(false);
            }
        }
    }
}
void VideoEngine::setEnStream(bool _enStream)
{
    m_enStream = _enStream;

}
bool VideoEngine::enStream()
{
    return m_enStream;
}
void VideoEngine::setEnSaving(bool _enSaving)
{
    m_enSaving = _enSaving;
}
bool VideoEngine::enSaving()
{
    return m_enSaving;
}
void VideoEngine::setSensorMode(bool _sensorMode)
{
    m_sensorMode = _sensorMode;
}
bool VideoEngine::sensorMode()
{
    return m_sensorMode;
}
int VideoEngine::frameID(){
    return m_frameID;
}
bool VideoEngine::enOD()
{
    return m_enOD;
}
bool VideoEngine::enTrack()
{
    return m_enTrack;
}
bool VideoEngine::enSteer()
{
    return m_enSteer;
}

QMap<int,bool>  VideoEngine::freezeMap(){ return m_freezeMap; }
void VideoEngine::setSourceRTSP(QString source, int port, int width, int height){
    printf("VideoEngine::setSourceRTSP source=%s\r\n",source.toStdString().c_str());
    stopRTSP();
    m_vRTSPServer = new VRTSPServer();
    m_vRTSPServer->m_source = source;
    m_vRTSPServer->m_port = port;
    m_vRTSPServer->m_width = width;
    m_vRTSPServer->m_height = height;
    m_vRTSPServer->start();
}
void VideoEngine::stopRTSP(){
    if(m_vRTSPServer!=nullptr && !(m_vRTSPServer->m_stop)){
        m_vRTSPServer->m_stop = true;
        m_vRTSPServer->setStateRun(false);
        m_vRTSPServer->wait(1000);
        m_vRTSPServer->quit();
        if (!m_vRTSPServer->wait(1000)) {
            m_vRTSPServer->terminate();
            m_vRTSPServer->wait(1000);
        }
        delete m_vRTSPServer;
        m_vRTSPServer = nullptr;
    }
}
void VideoEngine::slTrackStateLost(){
    m_gimbal->setGimbalRate(0,0);
    m_gimbal->context()->m_lockMode = "FREE";
    removeTrackObjectInfo(0);
    Q_EMIT trackStateLost();
}
void VideoEngine::slTrackStateFound(int _id, double _px, double _py, double _oW, double _oH, double _w, double _h){
//    printf("%s]\r\n",__func__);
    if(m_gimbal != nullptr){
        m_gimbal->lockScreenPoint(_id,_px,_py,_oW,_oH,_w,_h);
    }
    updateTrackObjectInfo("Object","RECT",QVariant(QRect(
                                                       static_cast<int>(_px-_oW/2),
                                                       static_cast<int>(_py-_oH/2),
                                                       static_cast<int>(_oW),
                                                       static_cast<int>(_oH)))
                          );
    updateTrackObjectInfo("Object","LATITUDE",QVariant(20.975092+_px/1000000));
    updateTrackObjectInfo("Object","LONGIITUDE",QVariant(105.307680+_py/1000000));
    updateTrackObjectInfo("Object","SPEED",QVariant(_py));
    updateTrackObjectInfo("Object","ANGLE",QVariant(_px));
    Q_EMIT trackStateFound(_id,_px,_py,_oW, _oH, _w, _h);
}
int VideoEngine::addVideoRender(VideoRender *viewer){
    printf("%s[%d] %p\r\n",__func__,m_listRender.size(),viewer);
    this->freezeMap()[m_listRender.size()] = true;
    m_listRender.append(viewer);
    return m_listRender.size() - 1;
}
void VideoEngine::removeVideoRender(int viewerID){
     printf("%s[%d] %d\r\n",__func__,m_listRender.size(),viewerID);
    if(viewerID >= 0 && viewerID < m_listRender.size()){
        this->freezeMap().remove(viewerID);
        m_listRender.removeAt(viewerID);
    }
}

void VideoEngine::drawOnRenderID(int viewerID, unsigned char *data, int width, int height,float* warpMatrix, unsigned char *dataOut){
    if(viewerID >=0 && viewerID < m_listRender.size()){
        VideoRender* tmpViewer = m_listRender[viewerID];
        if(tmpViewer != nullptr){
            if((width != m_sourceSize.width() ||
                height != m_sourceSize.height()) &&viewerID == 0){
                Q_EMIT VideoEngine::sourceSizeChanged(width,height);
            }
            tmpViewer->handleNewFrame(viewerID,data,width,height,warpMatrix,dataOut);

//            printf("%s[%d] dataOut=%p\r\n",__func__,viewerID,dataOut);
        }
    }
}
QSize VideoEngine::sourceSize()
{
    return m_sourceSize;
}
void VideoEngine::onStreamFrameSizeChanged(int width, int height)
{
//    printf("%s [%dx%d]\r\n",__func__,width,height);
    m_vSavingWorker->setStreamSize(width, height);
    if (m_enStream) {
        if(m_vRTSPServer == nullptr)
        {
    #ifdef USE_VIDEO_CPU
        setSourceRTSP("( appsrc name=othersrc ! avenc_mpeg4 bitrate=4000000 ! rtpmp4vpay config-interval=3 name=pay0 pt=96 )",
                      8554,width,height);
    #endif
    #ifdef USE_VIDEO_GPU
        setSourceRTSP("( appsrc name=othersrc ! nvh264enc bitrate=4000000 ! h264parse ! rtph264pay mtu=1400 name=pay0 pt=96 )",
                      8554,width,height);
    #endif
        }else{
//            printf("m_vRTSPServer != nullptr\r\n");
        }
    }

    if (m_enSaving) {
        m_vSavingWorker->start();
    }
}
void VideoEngine::rectangle(cv::Mat& imgY,cv::Mat& imgU,cv::Mat& imgV,cv::Rect rect,cv::Scalar color,int thickness, int lineType,int shift){
    double y,u,v;
    convertRGB2YUV((double)color.val[0],(double)color.val[1],(double)color.val[2],y,u,v);

    cv::rectangle(imgY,rect,cv::Scalar(y),thickness,lineType,shift);
    cv::rectangle(imgU,cv::Rect(rect.x/2,rect.y/2, rect.width/2, rect.height/2),cv::Scalar(u),thickness/2,lineType,shift);
    cv::rectangle(imgV,cv::Rect(rect.x/2,rect.y/2, rect.width/2, rect.height/2),cv::Scalar(v),thickness/2,lineType,shift);
}

void VideoEngine::fillRectangle(cv::Mat& imgY,cv::Mat& imgU,cv::Mat& imgV,cv::Rect rect,cv::Scalar color){
    double y,u,v;
    convertRGB2YUV((double)color.val[0],(double)color.val[1],(double)color.val[2],y,u,v);

    cv::Mat modeBackgroundY(rect.height,rect.width,CV_8UC1,cv::Scalar(y));
    cv::Mat modeBackgroundU(rect.height/2, rect.width/2, CV_8UC1,cv::Scalar(u));
    cv::Mat modeBackgroundV(rect.height/2, rect.width/2, CV_8UC1,cv::Scalar(v));

    modeBackgroundY.copyTo(imgY(rect));
    modeBackgroundU.copyTo(imgU(cv::Rect(rect.x/2,rect.y/2, rect.width/2, rect.height/2)));
    modeBackgroundV.copyTo(imgV(cv::Rect(rect.x/2,rect.y/2, rect.width/2, rect.height/2)));
}

void VideoEngine::line(cv::Mat& imgY,cv::Mat& imgU,cv::Mat& imgV,cv::Point start,cv::Point stop,cv::Scalar color,
          int thickness,
          int lineType, int shift){
    double y,u,v;
    convertRGB2YUV((double)color.val[0],(double)color.val[1],(double)color.val[2],y,u,v);

    cv::line(imgY,start,stop,cv::Scalar(y),thickness,lineType,shift);
    cv::line(imgU,cv::Point(start.x/2,start.y/2),cv::Point(stop.x/2,stop.y/2),cv::Scalar(u),thickness/2,lineType,shift);
    cv::line(imgV,cv::Point(start.x/2,start.y/2),cv::Point(stop.x/2,stop.y/2),cv::Scalar(v),thickness/2,lineType,shift);
}

void VideoEngine::putText(cv::Mat& imgY,cv::Mat& imgU,cv::Mat& imgV,const string& text, cv::Point org,
             int fontFace, double fontScale, cv::Scalar color,
             int thickness, int lineType,
             bool bottomLeftOrigin){
    double y,u,v;
    convertRGB2YUV((double)color.val[0],(double)color.val[1],(double)color.val[2],y,u,v);

    cv::putText(imgY,text, org, fontFace, fontScale, cv::Scalar(y), thickness,lineType,bottomLeftOrigin);
    cv::putText(imgU,text, cv::Point(org.x /2,org.y/2), fontFace, fontScale/2, cv::Scalar(u), thickness/2,lineType,bottomLeftOrigin);
    cv::putText(imgV,text, cv::Point(org.x /2,org.y/2), fontFace, fontScale/2, cv::Scalar(v), thickness/2,lineType,bottomLeftOrigin);
}

void VideoEngine::ellipse(cv::Mat& imgY,cv::Mat& imgU,cv::Mat& imgV,cv::Point center,
             cv::Size size,double angle,double startAngle,double endAngle,
             cv::Scalar color,int thickness, int lineType, int shift){
    double y,u,v;
    convertRGB2YUV((double)color.val[0],(double)color.val[1],(double)color.val[2],y,u,v);
    cv::ellipse(imgY,center,size,angle,startAngle,endAngle,cv::Scalar(y),thickness,lineType,shift);
    cv::ellipse(imgU,cv::Point(center.x/2,center.y/2),
                cv::Size(size.width/2,size.height/2),
                angle,startAngle,endAngle,cv::Scalar(u),thickness/2,lineType,shift);
    cv::ellipse(imgV,cv::Point(center.x/2,center.y/2),
                cv::Size(size.width/2,size.height/2),
                angle,startAngle,endAngle,cv::Scalar(v),thickness/2,lineType,shift);
}
void VideoEngine::convertRGB2YUV(const double R, const double G, const double B, double& Y, double& U, double& V)
{
    Y =  0.257 * R + 0.504 * G + 0.098 * B +  16;
    U = -0.148 * R - 0.291 * G + 0.439 * B + 128;
    V =  0.439 * R - 0.368 * G - 0.071 * B + 128;
}
