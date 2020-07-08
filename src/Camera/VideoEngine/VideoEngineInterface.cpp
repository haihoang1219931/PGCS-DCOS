#include "VideoEngineInterface.h"
#include "../VideoDisplay/VideoRender.h"
#include "VRTSPServer.h"
#include "VSavingWorker.h"
#include "../Cache/TrackObject.h"
#include "Joystick/JoystickLib/JoystickThreaded.h"
#include "../GimbalController/GimbalInterface.h"
#include "Camera/GimbalController/GimbalInterface.h"
#include "Bytes/ByteManipulation.h"
#ifndef rad2Deg
#define rad2Deg 57.2957795f
#endif
VideoEngine::VideoEngine(QObject *parent) : QObject(parent)
{
    m_targetLocation = new TargetLocalization();
    m_targetLocation->visionViewInit(0.006f,1920,1080);
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
        if(m_targetLocation!= nullptr){
            double uavPosition[2];
            uavPosition[0] = m_gimbal->context()->m_latitude;
            uavPosition[1] = m_gimbal->context()->m_longitude;
            double center[2];
            m_targetLocation->targetLocationMain(
                        _px,_py,
                        m_gimbal->context()->m_hfov[m_gimbal->context()->m_sensorID] / rad2Deg,
                        m_gimbal->context()->m_rollOffset / rad2Deg,
                        m_gimbal->context()->m_pitchOffset / rad2Deg,
                        m_gimbal->context()->m_yawOffset / rad2Deg,
                        m_gimbal->context()->m_panPosition / rad2Deg,
                        m_gimbal->context()->m_tiltPosition / rad2Deg,
                        uavPosition[0],
                        uavPosition[1],
                        m_gimbal->context()->m_altitudeOffset,
                        0,
                        center
                    );
            updateTrackObjectInfo("Object","RECT",QVariant(QRect(
                                                               static_cast<int>(_px-_oW/2),
                                                               static_cast<int>(_py-_oH/2),
                                                               static_cast<int>(_oW),
                                                               static_cast<int>(_oH)))
                                  );
            updateTrackObjectInfo("Object","LATITUDE",QVariant(center[0]));
            updateTrackObjectInfo("Object","LONGIITUDE",QVariant(center[1]));
            updateTrackObjectInfo("Object","SPEED",QVariant(_py));
            updateTrackObjectInfo("Object","ANGLE",QVariant(_px));
        }

    }

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
                //                printf("drawOnRenderID (%dx%d)\r\n",width,height);
                m_sourceSize.setWidth(width);
                m_sourceSize.setHeight(height);
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
        }
    }

    if (m_enSaving) {
        m_vSavingWorker->setStreamSize(width, height);
        m_vSavingWorker->start();
    }
}
void VideoEngine::drawSteeringCenter(cv::Mat &imgY,cv::Mat &imgU,cv::Mat &imgV,
                                     int _wBoundary, int _centerX, int _centerY,
                                     cv::Scalar _color)
{
    _centerX -= _wBoundary / 2;
    _centerY -= _wBoundary / 2;
    VideoEngine::line(imgY,imgU,imgV, cv::Point(_centerX, _centerY),
                      cv::Point(_centerX + _wBoundary / 4, _centerY), _color, 2);
    VideoEngine::line(imgY,imgU,imgV, cv::Point(_centerX, _centerY),
                      cv::Point(_centerX, _centerY + _wBoundary / 4), _color, 2);
    VideoEngine::line(imgY,imgU,imgV, cv::Point(_centerX + _wBoundary, _centerY),
                      cv::Point(_centerX + 3 * _wBoundary / 4, _centerY), _color, 2);
    VideoEngine::line(imgY,imgU,imgV, cv::Point(_centerX + _wBoundary, _centerY),
                      cv::Point(_centerX + _wBoundary, _centerY + _wBoundary / 4), _color,
                      2);
    VideoEngine::line(imgY,imgU,imgV, cv::Point(_centerX, _centerY + _wBoundary),
                      cv::Point(_centerX, _centerY + 3 * _wBoundary / 4), _color, 2);
    VideoEngine::line(imgY,imgU,imgV, cv::Point(_centerX, _centerY + _wBoundary),
                      cv::Point(_centerX + _wBoundary / 4, _centerY + _wBoundary), _color,
                      2);
    VideoEngine::line(imgY,imgU,imgV, cv::Point(_centerX + _wBoundary, _centerY + _wBoundary),
                      cv::Point(_centerX + _wBoundary, _centerY + 3 * _wBoundary / 4),
                      _color, 2);
    VideoEngine::line(imgY,imgU,imgV, cv::Point(_centerX + _wBoundary, _centerY + _wBoundary),
                      cv::Point(_centerX + 3 * _wBoundary / 4, _centerY + _wBoundary),
                      _color, 2);
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

unsigned short VideoEngine::checksum(
        unsigned char * buff, // Pointer to the first byte in the 16-byte UAS Datalink LS key.
        unsigned short len ) //Length from 16-byte US key up to 1-byte checksum length.
{
    unsigned short bcc = 0, i; // Initialize Checksum and counter variables.
    for ( i = 0 ; i < len; i++)
        bcc += buff[i] << (8 * ((i + 1) % 2));
    return bcc;
} // end of bcc_16 ()
std::vector<uint8_t> VideoEngine::encodeMeta(GimbalInterface* gimbal){
    uint8_t keyST0601[] = {0X06,0X0E,0X2B,0X34,0X02,0X0B,0X01,0X01,0X0E,0X01,0X03,0X01,0X01,0X00,0X00,0X00};
    int totalLength = 0;
    uint16_t checkSum = 0;
    struct timeval tv;
    gettimeofday(&tv,NULL);
    uint64_t timestamp = (1000000*tv.tv_sec) + tv.tv_usec;
    printf("");
//    timespec ts;
//    timespec_get(&ts,TIME_UTC);
//    uint64_t timestamp = ts.tv_sec * 1000000 + ts.tv_nsec/1000;


//    printf();
//    int localHour,gmtHour;
//    time_t t = time(0);
//    struct tm * lcl = localtime(&t);
//    localHour = lcl->tm_hour;

//    struct tm * gmt = gmtime(&t);
//    gmtHour = gmt->tm_hour;
//    uint64_t timestamp = (t + (localHour-gmtHour)*3600)*1000000+ ts.tv_nsec/1000;
    std::string missionID = gimbal->context()->m_missionID.toStdString();
    std::string platformTailNumber = gimbal->context()->m_platformTailNumber.toStdString();
    float m_platformHeadingAngle = gimbal->context()->m_yawOffset;
    float m_platformPitchAngle = gimbal->context()->m_pitchOffset;
    float m_platformRollAngle = gimbal->context()->m_rollOffset;
    std::string m_platformDesignation = gimbal->context()->m_platformDesignation.toStdString();
    float m_sensorLatitude = gimbal->context()->m_latitude;
    float m_sensorLongitude = gimbal->context()->m_longitude;
    float m_sensorAltitude = gimbal->context()->m_altitudeOffset;
    float m_sensorHFOV = gimbal->context()->m_hfov[gimbal->context()->m_sensorID];
    float m_sensorVFOV = gimbal->context()->m_vfov[gimbal->context()->m_sensorID];
    float m_sensorRelativeAzimuth = gimbal->context()->m_panPosition;
    float m_sensorRelativeElevation = gimbal->context()->m_tiltPosition;
    float m_sensorRelativeRoll = gimbal->context()->m_rollOffset;
    float m_slantRanged = gimbal->context()->m_targetSlr;
    float m_frameCenterLatitude = gimbal->context()->m_centerLat;
    float m_frameCenterLongitude = gimbal->context()->m_centerLon;
    float m_frameCenterElevation = gimbal->context()->m_centerAlt;
    float m_offsetCornerLatitudePoint1 = -gimbal->context()->m_cornerLat[0] + gimbal->context()->m_centerLat;
    float m_offsetCornerLongitudePoint1 = -gimbal->context()->m_cornerLon[0] + gimbal->context()->m_centerLon;
    float m_offsetCornerLatitudePoint2 = -gimbal->context()->m_cornerLat[1] + gimbal->context()->m_centerLat;
    float m_offsetCornerLongitudePoint2 = -gimbal->context()->m_cornerLon[1] + gimbal->context()->m_centerLon;
    float m_offsetCornerLatitudePoint3 = -gimbal->context()->m_cornerLat[2] + gimbal->context()->m_centerLat;
    float m_offsetCornerLongitudePoint3 = -gimbal->context()->m_cornerLon[2] + gimbal->context()->m_centerLon;
    float m_offsetCornerLatitudePoint4 = -gimbal->context()->m_cornerLat[3] + gimbal->context()->m_centerLat;
    float m_offsetCornerLongitudePoint4 = -gimbal->context()->m_cornerLon[3] + gimbal->context()->m_centerLon;

    std::vector<uint8_t> metaData;

    Klv kTimeStamp(0x02,8,ByteManipulation::ToBytes(timestamp,Endianness::Little));
    metaData.insert(metaData.end(),kTimeStamp.m_encoded.begin(),kTimeStamp.m_encoded.end());

    Klv kMissionID(0x03,static_cast<uint8_t>(missionID.length()),(uint8_t*)missionID.c_str());
    metaData.insert(metaData.end(),kMissionID.m_encoded.begin(),kMissionID.m_encoded.end());

    Klv kPlatformTailNumber(0x04,static_cast<uint8_t>(platformTailNumber.length()),(uint8_t*)platformTailNumber.c_str());
    metaData.insert(metaData.end(),kPlatformTailNumber.m_encoded.begin(),kPlatformTailNumber.m_encoded.end());

    unsigned short platformHeading = static_cast<unsigned short>(m_platformHeadingAngle*(65536-1)/360.0f);
    Klv kPlatformHeadAngle(0x05,2,ByteManipulation::ToBytes(
                               platformHeading,Endianness::Little));
    metaData.insert(metaData.end(),kPlatformHeadAngle.m_encoded.begin(),kPlatformHeadAngle.m_encoded.end());

    unsigned short platformPitch = static_cast<unsigned short>((m_platformPitchAngle+20)*(65534)/40.0f-32767);
    Klv kPlatformPitchAngle(0x06,2,ByteManipulation::ToBytes(
                                platformPitch,Endianness::Little));
    metaData.insert(metaData.end(),kPlatformPitchAngle.m_encoded.begin(),kPlatformPitchAngle.m_encoded.end());

    unsigned short platformRoll = static_cast<unsigned short>((m_platformRollAngle+50)*(65534)/100.0f-32767);
    Klv kPlatformRollAngle(0x07,2,ByteManipulation::ToBytes(
                               platformRoll,Endianness::Little));
    metaData.insert(metaData.end(),kPlatformRollAngle.m_encoded.begin(),kPlatformRollAngle.m_encoded.end());

    Klv kDesignation(0x0A,static_cast<uint8_t>(m_platformDesignation.length()),(uint8_t*)m_platformDesignation.c_str());
    metaData.insert(metaData.end(),kDesignation.m_encoded.begin(),kDesignation.m_encoded.end());

    int sensorLatitude = static_cast<int>((m_sensorLatitude)*(4294967294)/180.0f);
    Klv kSensorLatitude(0x0D,4,ByteManipulation::ToBytes(
                            sensorLatitude,Endianness::Little));
    metaData.insert(metaData.end(),kSensorLatitude.m_encoded.begin(),kSensorLatitude.m_encoded.end());

    int sensorLongitude = static_cast<int>((m_sensorLongitude)*(4294967294)/360.0f);
    Klv kSensorLongitude(0x0E,4,ByteManipulation::ToBytes(
                             sensorLongitude,Endianness::Little));
    metaData.insert(metaData.end(),kSensorLongitude.m_encoded.begin(),kSensorLongitude.m_encoded.end());

    unsigned short sensorAltitude = static_cast<unsigned short>((m_sensorAltitude+900)*(65535)/19900.0f);
    Klv kSensorAltitude(0x0F,2,ByteManipulation::ToBytes(
                            sensorAltitude,Endianness::Little));
    metaData.insert(metaData.end(),kSensorAltitude.m_encoded.begin(),kSensorAltitude.m_encoded.end());

    unsigned short sensorHFOV = static_cast<unsigned short>((m_sensorHFOV)*(65535)/180.0f);
    Klv kSensorHFOV(0x10,2,ByteManipulation::ToBytes(
                        sensorHFOV,Endianness::Little));
    metaData.insert(metaData.end(),kSensorHFOV.m_encoded.begin(),kSensorHFOV.m_encoded.end());

    unsigned short sensorVFOV = static_cast<unsigned short>((m_sensorVFOV)*(65535)/180.0f);
    Klv kSensorVFOV(0x11,2,ByteManipulation::ToBytes(
                        sensorVFOV,Endianness::Little));
    metaData.insert(metaData.end(),kSensorVFOV.m_encoded.begin(),kSensorVFOV.m_encoded.end());

    unsigned int sensorAzimuth = static_cast<unsigned int>((m_sensorRelativeAzimuth)*(4294967295)/360.0f);
    Klv kSensorAzimuth(0x12,4,ByteManipulation::ToBytes(
                           sensorAzimuth,Endianness::Little));
    metaData.insert(metaData.end(),kSensorAzimuth.m_encoded.begin(),kSensorAzimuth.m_encoded.end());

    int sensorElevation = static_cast<int>((m_sensorRelativeElevation)*(4294967295)/360.0f);
    Klv kSensorElevation(0x13,4,ByteManipulation::ToBytes(
                             sensorElevation,Endianness::Little));
    metaData.insert(metaData.end(),kSensorElevation.m_encoded.begin(),kSensorElevation.m_encoded.end());

    unsigned int sensorRoll = static_cast<unsigned int>((m_sensorRelativeRoll)*(4294967295)/360.0f);
    Klv kSensorRoll(0x14,4,ByteManipulation::ToBytes(
                        sensorRoll,Endianness::Little));
    metaData.insert(metaData.end(),kSensorRoll.m_encoded.begin(),kSensorRoll.m_encoded.end());

    unsigned int slantRanged = static_cast<unsigned int>((m_slantRanged)*(4294967295)/5000000.0f);
    Klv kSlantRanged(0x15,4,ByteManipulation::ToBytes(
                         slantRanged,Endianness::Little));
    metaData.insert(metaData.end(),kSlantRanged.m_encoded.begin(),kSlantRanged.m_encoded.end());

    int frameCenterLatitude = static_cast<int>((m_frameCenterLatitude)*(4294967294)/180.0f);
    Klv kFrameCenterLatitude(0x17,4,ByteManipulation::ToBytes(
                                 frameCenterLatitude,Endianness::Little));
    metaData.insert(metaData.end(),kFrameCenterLatitude.m_encoded.begin(),kFrameCenterLatitude.m_encoded.end());

    int frameCenterLongitude = static_cast<int>((m_frameCenterLongitude)*(4294967294)/360.0f);
    Klv kFrameCenterLongitude(0x18,4,ByteManipulation::ToBytes(
                                  frameCenterLongitude,Endianness::Little));
    metaData.insert(metaData.end(),kFrameCenterLongitude.m_encoded.begin(),kFrameCenterLongitude.m_encoded.end());

    unsigned short frameCenterElevation = static_cast<unsigned short>((m_frameCenterElevation+900)/19900.0f*(65535.0f));
    Klv kFrameCenterElevation(0x19,2,ByteManipulation::ToBytes(
                                  frameCenterElevation,Endianness::Little));
    metaData.insert(metaData.end(),kFrameCenterElevation.m_encoded.begin(),kFrameCenterElevation.m_encoded.end());

    unsigned short offsetLatitudePoint1 = static_cast<unsigned short>((m_offsetCornerLatitudePoint1)*(65534.0f)/0.15f);
    Klv kOffsetLatitudePoint1(0x1A,2,ByteManipulation::ToBytes(
                                  offsetLatitudePoint1,Endianness::Little));
    metaData.insert(metaData.end(),kOffsetLatitudePoint1.m_encoded.begin(),kOffsetLatitudePoint1.m_encoded.end());

    unsigned short offsetLongitudePoint1 = static_cast<unsigned short>((m_offsetCornerLongitudePoint1)*(65534.0f)/0.15f);
    Klv kOffsetLongitudePoint1(0x1B,2,ByteManipulation::ToBytes(
                                   offsetLongitudePoint1,Endianness::Little));
    metaData.insert(metaData.end(),kOffsetLongitudePoint1.m_encoded.begin(),kOffsetLongitudePoint1.m_encoded.end());

    unsigned short offsetLatitudePoint2 = static_cast<unsigned short>((m_offsetCornerLatitudePoint2)*(65534.0f)/0.15f);
    Klv kOffsetLatitudePoint2(0x1C,2,ByteManipulation::ToBytes(
                                  offsetLatitudePoint2,Endianness::Little));
    metaData.insert(metaData.end(),kOffsetLatitudePoint2.m_encoded.begin(),kOffsetLatitudePoint2.m_encoded.end());

    unsigned short offsetLongitudePoint2 = static_cast<unsigned short>((m_offsetCornerLongitudePoint2)*(65534.0f)/0.15f);
    Klv kOffsetLongitudePoint2(0x1D,2,ByteManipulation::ToBytes(
                                   offsetLongitudePoint2,Endianness::Little));
    metaData.insert(metaData.end(),kOffsetLongitudePoint2.m_encoded.begin(),kOffsetLongitudePoint2.m_encoded.end());

    unsigned short offsetLatitudePoint3 = static_cast<unsigned short>((m_offsetCornerLatitudePoint3)*(65534.0f)/0.15f);
    Klv kOffsetLatitudePoint3(0x1E,2,ByteManipulation::ToBytes(
                                  offsetLatitudePoint3,Endianness::Little));
    metaData.insert(metaData.end(),kOffsetLatitudePoint3.m_encoded.begin(),kOffsetLatitudePoint3.m_encoded.end());

    unsigned short offsetLongitudePoint3 = static_cast<unsigned short>((m_offsetCornerLongitudePoint3)*(65534.0f)/0.15f);
    Klv kOffsetLongitudePoint3(0x1F,2,ByteManipulation::ToBytes(
                                   offsetLongitudePoint3,Endianness::Little));
    metaData.insert(metaData.end(),kOffsetLongitudePoint3.m_encoded.begin(),kOffsetLongitudePoint3.m_encoded.end());

    unsigned short offsetLatitudePoint4 = static_cast<unsigned short>((m_offsetCornerLatitudePoint4)*(65534.0f)/0.15f);
    Klv kOffsetLatitudePoint4(0x20,2,ByteManipulation::ToBytes(
                                  offsetLatitudePoint4,Endianness::Little));
    metaData.insert(metaData.end(),kOffsetLatitudePoint4.m_encoded.begin(),kOffsetLatitudePoint4.m_encoded.end());

    unsigned short offsetLongitudePoint4 = static_cast<unsigned short>((m_offsetCornerLongitudePoint4)*(65534.0f)/0.15f);
    Klv kOffsetLongitudePoint4(0x21,2,ByteManipulation::ToBytes(
                                   offsetLongitudePoint4,Endianness::Little));
    metaData.insert(metaData.end(),kOffsetLongitudePoint4.m_encoded.begin(),kOffsetLongitudePoint4.m_encoded.end());

    unsigned int dataLength = metaData.size()+4;
    if(dataLength < 128){
        uint8_t length = dataLength;
        metaData.insert(metaData.begin(),length);
    }else{
        uint16_t length = (uint16_t)dataLength | 0x8100;
        std::vector<uint8_t> aLength = ByteManipulation::ToBytes(length,Endianness::Little);
        metaData.insert(metaData.begin(),aLength.begin(),aLength.end());
    }
    std::vector<uint8_t> kKeyST0601(keyST0601,keyST0601+sizeof(keyST0601));
    metaData.insert(metaData.begin(),kKeyST0601.begin(),kKeyST0601.end());

    metaData.push_back(0X01);
    metaData.push_back(0X02);
    checkSum = checksum(metaData.data(),metaData.size());
    std::vector<uint8_t> kChecksum = ByteManipulation::ToBytes(checkSum,Endianness::Little);
    metaData.insert(metaData.end(),kChecksum.begin(),kChecksum.end());

    return metaData;
}
