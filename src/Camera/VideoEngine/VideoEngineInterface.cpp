#include "VideoEngineInterface.h"
#include "../VideoDisplay/ImageItem.h"
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
void VideoEngine::updateVideoSurface(int width, int height){
    m_updateCount = 0;
    m_updateVideoSurface = true;
    m_videoSurfaceSize.setWidth(width);
    m_videoSurfaceSize.setHeight(height);
}
int VideoEngine::addSubViewer(ImageItem *viewer){
    printf("%s[%d] %p\r\n",__func__,m_listSubViewer.size(),viewer);
    this->freezeMap()[m_listSubViewer.size()] = true;
    m_listSubViewer.append(viewer);
    return m_listSubViewer.size() -1;
}
void VideoEngine::removeSubViewer(int viewerID){
     printf("%s[%d] %d\r\n",__func__,m_listSubViewer.size(),viewerID);
    if(viewerID >= 0 && viewerID < m_listSubViewer.size()){
        this->freezeMap().remove(viewerID);
        m_listSubViewer.removeAt(viewerID);
    }
}
QAbstractVideoSurface *VideoEngine::videoSurface()
{
    return m_videoSurface;
}
void VideoEngine::drawOnViewerID(cv::Mat img, int viewerID){
    if(viewerID >=0 && viewerID < m_listSubViewer.size()){
        ImageItem* tmpViewer = m_listSubViewer[viewerID];
        if(tmpViewer != nullptr){
            if(viewerID == 0){
                if(m_sourceSize.width() != img.cols ||
                        m_sourceSize.height() != img.rows){
                    m_sourceSize.setWidth(img.cols);
                    m_sourceSize.setHeight(img.rows);
//                    sourceSizeChanged(img.cols,img.rows);
                }
            }
            cv::Mat imgDraw;
            cv::Mat imgResize;
            cv::resize(img,imgResize,cv::Size(tmpViewer->boundingRect().width(),tmpViewer->boundingRect().height()));

            if(imgResize.channels() == 3){
                cv::cvtColor(imgResize,imgDraw,cv::COLOR_BGR2RGBA);
            }else if(imgResize.channels() == 4){
                cv::cvtColor(imgResize,imgDraw,cv::COLOR_RGBA2BGRA);
            }
            imageDataMutex[viewerID].lock();
            memcpy(imageData[viewerID],imgDraw.data,imgDraw.cols*imgDraw.rows*4);
            QImage tmp((uchar*)imageData[viewerID], imgDraw.cols, imgDraw.rows, QImage::Format_RGBA8888);
            tmpViewer->setImage(tmp);
            imageDataMutex[viewerID].unlock();
        }else{
        }
    }
}

void VideoEngine::setVideoSurface(QAbstractVideoSurface *_videoSurface)
{
    printf("setVideoSurface");

    if (m_videoSurface != _videoSurface) {
        m_videoSurface = _videoSurface;
        update();
    }
}
void VideoEngine::update()
{
    printf("Update video surface(%d,%d)\r\n",
           m_videoSurfaceSize.width(),
           m_videoSurfaceSize.height());
    if (m_videoSurface) {
        if (m_videoSurface->isActive()) {
            m_videoSurface->stop();
        }
        if (!m_videoSurface->start(QVideoSurfaceFormat(m_videoSurfaceSize, VIDEO_OUTPUT_FORMAT))) {
            printf("Could not start QAbstractVideoSurface, error: %d", m_videoSurface->error());
        } else {
            printf("Start QAbstractVideoSurface done\r\n");
        }
    }
}
QSize VideoEngine::sourceSize()
{
    return m_sourceSize;
}
void VideoEngine::onStreamFrameSizeChanged(int width, int height)
{
    printf("%s [%dx%d]\r\n",__func__,width,height);
    m_vSavingWorker->setStreamSize(width, height);
    if (m_enStream) {
        if(m_vRTSPServer == nullptr)
        {
    #ifdef USE_VIDEO_CPU
        setSourceRTSP("( appsrc name=othersrc ! avenc_mpeg4 bitrate=1500000 ! rtpmp4vpay config-interval=3 name=pay0 pt=96 )",
                      8554,width,height);
    #endif
    #ifdef USE_VIDEO_GPU
//        setSourceRTSP("( appsrc name=othersrc ! nvh264enc bitrate=1500000 ! h264parse ! rtph264pay mtu=1400 name=pay0 pt=96 )",
//                      8554,width,height);
        setSourceRTSP("( appsrc name=othersrc ! videoscale ! video/x-raw,width=1280,height=720 ! avenc_mpeg4 bitrate=2000000 ! rtpmp4vpay config-interval=3 name=pay0 pt=96 )",
                      8554,width,height);
    #endif
        }else{
            printf("m_vRTSPServer != nullptr\r\n");
        }
    }

    if (m_enSaving) {
        m_vSavingWorker->start();
    }
}
