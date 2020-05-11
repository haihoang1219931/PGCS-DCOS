#include "VDisplay.h"
#include "src/Camera/GimbalController/GimbalInterface.h"
VDisplay::VDisplay(VideoEngine *_parent) : VideoEngine(_parent)
{
    m_frameID = -1;
    m_vFrameGrabber = new VFrameGrabber;
    m_vPreprocess = new VPreprocess;
    m_vODWorker = new VODWorker;
    m_vMOTWorker = new VMOTWorker;
    m_vSearchWorker = new VSearchWorker;
    m_vTrackWorker = new VTrackWorker;
    m_vRTSPServer = new VRTSPServer;
    m_vSavingWorker = new VSavingWorker("EO");
    m_threadEODisplay = new QThread(0);
    m_vDisplayWorker = new VDisplayWorker(0);
    m_vDisplayWorker->moveToThread(m_threadEODisplay);
    connect(m_vDisplayWorker, SIGNAL(receivedFrame(int, QVideoFrame)), this,
            SLOT(onReceivedFrame(int, QVideoFrame)));
    connect(m_vDisplayWorker, SIGNAL(receivedFrame()), this,
            SLOT(onReceivedFrame()));
    connect(m_threadEODisplay, SIGNAL(started()), m_vDisplayWorker,
            SLOT(process()));
    connect(m_vTrackWorker, &VTrackWorker::trackStateFound,
            this, &VDisplay::slDeterminedTrackObjected);
    connect(m_vTrackWorker, SIGNAL(determinedPlateOnTracking(QString, QString)),
            this, SIGNAL(determinedPlateOnTracking(QString, QString)));
    connect(m_vTrackWorker, &VTrackWorker::objectLost,
            this, &VDisplay::slObjectLost);
    connect(m_vDisplayWorker,&VDisplayWorker::readyDrawOnViewerID,this,&VDisplay::drawOnViewerID);
    m_videoSurfaceSize.setWidth(-1);
    m_videoSurfaceSize.setHeight(-1);
    init();
}

VDisplay::~VDisplay()
{
    this->stop();
    m_vRTSPServer->deleteLater();
    m_vFrameGrabber->deleteLater();
    m_vSavingWorker->deleteLater();
    m_vODWorker->deleteLater();
    m_vMOTWorker->deleteLater();
    m_vSearchWorker->deleteLater();
    m_vTrackWorker->deleteLater();
    m_vPreprocess->deleteLater();
}
void VDisplay::setGimbal(GimbalInterface* gimbal){
    m_gimbal = gimbal;
    m_vTrackWorker->m_gimbal = gimbal;
}
void VDisplay::init()
{
    std::string names_file   = "../GPUBased/OD/yolo-setup/visdrone2019.names";
    std::string cfg_file     = "../GPUBased/OD/yolo-setup/yolov3-tiny_3l.cfg";
    std::string weights_file = "../GPUBased/OD/yolo-setup/yolov3-tiny_3l_last.weights";
    std::string plate_cfg_file_click = "../GPUBased/Clicktrack/yolo-setup/yolov3-tiny_512.cfg";
    std::string plate_weights_file_click = "../GPUBased/Clicktrack/yolo-setup/yolov3-tiny_best.weights";
    std::string plate_cfg_search = "../GPUBased/plateOCR/yolo-setup/yolov3-tiny.cfg";
    std::string plate_weights_search = "../GPUBased/plateOCR/yolo-setup/yolov3-tiny_best.weights";
    m_detector = new Detector(cfg_file, weights_file);
    m_vODWorker->setDetector(m_detector);
    m_clicktrackDetector = new Detector(plate_cfg_file_click, plate_weights_file_click);
    m_vTrackWorker->setClicktrackDetector(m_clicktrackDetector);

    m_searchDetector = new Detector(plate_cfg_search, plate_weights_search);
    m_vSearchWorker->setPlateDetector(m_searchDetector);

    m_OCR = new OCR();
    m_vTrackWorker->setOCR(m_OCR);
    m_vSearchWorker->setOCR(m_OCR);
}

void VDisplay::start()
{
    m_vFrameGrabber->start();
    m_vPreprocess->start();
    m_vODWorker->start();
    m_vMOTWorker->start();
    m_vSearchWorker->start();
    m_vRTSPServer->start();
    m_vSavingWorker->start();
    m_vTrackWorker->start();
    m_threadEODisplay->start();
}


void VDisplay::onReceivedFrame(int _id, QVideoFrame frame)
{
    if(m_videoSurface!=nullptr){
        m_frameID = _id;
        if (m_sourceSize.width() != frame.width() ||
                m_sourceSize.height() != frame.height()) {
            m_sourceSize.setWidth(frame.width());
            m_sourceSize.setHeight(frame.height());
            Q_EMIT sourceSizeChanged(frame.width(),frame.height());
        }
        if(m_updateVideoSurface){
            if(m_updateCount < m_updateMax){
                update();
                m_updateCount ++;
            }else
                m_updateVideoSurface = false;
        }
        m_videoSurface->present(frame);

    }
}
void VDisplay::onReceivedFrame()
{
    if(m_videoSurface!=nullptr){
        m_frameID = m_vDisplayWorker->m_currID;
        if(m_updateVideoSurface){
            if(m_updateCount < m_updateMax){
                update();
                m_updateCount ++;
            }else
                m_updateVideoSurface = false;
        }
        QVideoFrame frame = QVideoFrame(
                    QImage((uchar *)m_vDisplayWorker->m_imgShow.data,
                           m_vDisplayWorker->m_imgShow.cols,
                           m_vDisplayWorker->m_imgShow.rows, QImage::Format_RGBA8888));
        frame.map(QAbstractVideoBuffer::ReadOnly);
        if (m_sourceSize.width() != frame.width() ||
                m_sourceSize.height() != frame.height()) {
            m_sourceSize.setWidth(frame.width());
            m_sourceSize.setHeight(frame.height());
            Q_EMIT sourceSizeChanged(frame.width(), frame.height());
        }

        m_videoSurface->present(frame);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        frame.unmap();
    }
}
void VDisplay::setVideo(QString _ip, int _port)
{
    printf("%s - %s\r\n",__func__,_ip.toStdString().c_str());
    m_vFrameGrabber->setSource(_ip.toStdString(), _port);
}
void VDisplay::setObjectDetect(bool enable){
    if(enable){
        std::vector<int> classIDs;
        for (int i = 0; i < 12; i++) {
            classIDs.push_back(i);
        }
        m_vDisplayWorker->setListObjClassID(classIDs);
        m_enOD = true;
        m_vODWorker->enableOD();
        m_vMOTWorker->enableMOT();
        // this line should be remove in the final product
        m_vSearchWorker->enableSearch();
    }else{
        m_enOD = false;
        m_vODWorker->disableOD();
        m_vMOTWorker->disableMOT();
        // this line should be remove in the final product
        m_vSearchWorker->disableSearch();
    }
}
void VDisplay::setPowerLineDetect(bool enable){
    m_enPD = enable;
}
void VDisplay::searchByClass(QVariantList _classList)
{
    std::vector<int> classIDs;

    if (_classList.size() == 0) {
        m_vODWorker->disableOD();
        m_vMOTWorker->disableMOT();
        m_vSearchWorker->disableSearch();
    }

    for (int i = 0; i < _classList.size(); i++) {
        classIDs.push_back(_classList.at(i).toInt());
    }

    m_vDisplayWorker->setListObjClassID(classIDs);
    m_enOD = true;
    m_vODWorker->enableOD();
    m_vMOTWorker->enableMOT();
    m_vSearchWorker->enableSearch();
}

void VDisplay::disableObjectDetect()
{
    m_enOD = false;
    m_vODWorker->disableOD();
    m_vMOTWorker->disableMOT();
    // this line should be remove in the final product
    m_vSearchWorker->disableSearch();
}

void VDisplay::enableObjectDetect()
{
    std::vector<int> classIDs;

    for (int i = 0; i < 12; i++) {
        classIDs.push_back(i);
    }

    m_vDisplayWorker->setListObjClassID(classIDs);
    m_enOD = true;
    m_vODWorker->enableOD();
    m_vMOTWorker->enableMOT();
    // this line should be remove in the final product
    m_vSearchWorker->enableSearch();
}
void VDisplay::moveImage(float panRate,float tiltRate,float zoomRate,float alpha){
    m_vTrackWorker->moveImage(panRate,tiltRate,zoomRate,alpha);
}
void VDisplay::setTrackAt(int _id, double _px, double _py, double _w, double _h)
{
    if(m_gimbal != nullptr){
        if(m_gimbal->context()->m_lockMode == "FREE"){
            m_gimbal->context()->m_lockMode = "TRACK";
            m_vTrackWorker->m_trackEnable = true;
            m_enTrack = true;
            m_enSteer = false;
            int x = static_cast<int>(_px/_w*m_sourceSize.width());
            int y = static_cast<int>(_py/_h*m_sourceSize.height());
            printf("%s at (%dx%d)\r\n",__func__,x,y);
            removeTrackObjectInfo(0);
            TrackObjectInfo *object = new TrackObjectInfo(m_sourceSize,QRect(x-20,y-20,40,40),"Object",20.975092,105.307680,0,0,"Track");
            object->setIsSelected(true);
            addTrackObjectInfo(object);
        }
    }
    m_vTrackWorker->setClick(_px, _py, _w, _h);
}

void VDisplay::setVideoSavingState(bool _state)
{
//    m_vDisplayWorker->setVideoSavingState(_state);
    m_vFrameGrabber->setVideoSavingState(_state);
}

void VDisplay::setStab(bool _en)
{
    m_vDisplayWorker->setDigitalStab(_en);
}

void VDisplay::setRecord(bool _en)
{
    m_vDisplayWorker->setVideoSavingState(_en);
}
void VDisplay::setShare(bool enable)
{
    m_vDisplayWorker->m_enShare = enable;
}
void VDisplay::stop()
{
    printf("\nSTOP===============================================================");
    m_vRTSPServer->stopPipeline();
    m_vFrameGrabber->stopPipeline();
    m_vSavingWorker->stopPipeline();
    m_vODWorker->stop();
    m_vSearchWorker->stop();
    m_vMOTWorker->stop();
    m_vTrackWorker->stop();
    m_vPreprocess->stop();
    std::this_thread::sleep_for(std::chrono::seconds(1));
}
void VDisplay::capture(){
    m_vDisplayWorker->capture();
}
void VDisplay::changeTrackSize(int _val)
{
    m_vTrackWorker->changeTrackSize(_val);
}
