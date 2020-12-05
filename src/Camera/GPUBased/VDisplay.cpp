#include "VDisplay.h"
#include "src/Camera/GimbalController/GimbalInterface.h"
VDisplay::VDisplay(VideoEngine *_parent) : VideoEngine(_parent)
{
    m_enSaving = false;
    m_frameID = -1;
    m_vFrameGrabber = new VFrameGrabber;
    m_vPreprocess = new VPreprocess;
    m_vODWorker = new VODWorker;
    m_vMOTWorker = new VMOTWorker;
    m_vSearchWorker = new VSearchWorker;
    m_vTrackWorker = new VTrackWorker;
    m_vSavingWorker = new VSavingWorker();
    m_threadEODisplay = new QThread(0);
    m_vDisplayWorker = new VDisplayWorker(0);
    m_vDisplayWorker->moveToThread(m_threadEODisplay);
    m_vFrameGrabber->m_enSaving = &m_enSaving;
    connect(m_threadEODisplay, SIGNAL(started()), m_vDisplayWorker,
            SLOT(process()));
    connect(m_vTrackWorker, &VTrackWorker::trackStateFound,
            this, &VDisplay::slTrackStateFound);
    connect(m_vTrackWorker, &VTrackWorker::determinedPlateOnTracking,
            this, &VideoEngine::determinedPlateOnTracking);
    connect(m_vTrackWorker, &VTrackWorker::trackStateLost,
            this, &VDisplay::slTrackStateLost);
    connect(m_vDisplayWorker,&VDisplayWorker::readyDrawOnRenderID,
            this,&VideoEngine::drawOnRenderID);
    connect(this, &VideoEngine::sourceSizeChanged,
            this, &VideoEngine::onStreamFrameSizeChanged);
    connect(m_vTrackWorker, &VTrackWorker::zoomCalculateChanged, this,&VDisplay::handleZoomCalculateChanged);
    connect(m_vTrackWorker, &VTrackWorker::zoomTargetChanged, this,&VDisplay::handleZoomTargetChanged);
    //    connect(m_vTrackWorker, &VTrackWorker::zoomTargetChangeStopped, this,&VDisplay::handleZoomTargetChangeStopped);
    m_videoSurfaceSize.setWidth(-1);
    m_videoSurfaceSize.setHeight(-1);
    init();
}

VDisplay::~VDisplay()
{
    this->stop();
    if(m_vFrameGrabber != nullptr)
        m_vFrameGrabber->deleteLater();
    m_vSavingWorker->deleteLater();
    m_vODWorker->deleteLater();
    m_vMOTWorker->deleteLater();
    m_vSearchWorker->deleteLater();
    m_vTrackWorker->deleteLater();
    m_vPreprocess->deleteLater();
}
void VDisplay::setdigitalZoom(float value){
    if(value >= 1 &&
            value <=8)
        m_vTrackWorker->m_zoomIR = value;
}
void VDisplay::setGimbal(GimbalInterface* gimbal){
    m_gimbal = gimbal;
    m_vTrackWorker->m_gimbal = gimbal;
    m_vFrameGrabber->m_gimbal = gimbal;
    m_vDisplayWorker->m_gimbal = gimbal;
}
void VDisplay::handleZoomTargetChangeStopped(float zoomTarget){
    if(m_gimbal!= nullptr){
        m_gimbal->setEOZoom("",zoomTarget);
    }
}
void VDisplay::handleZoomCalculateChanged(int index,float zoomCalculate){
    if(m_gimbal!= nullptr){
        m_gimbal->setZoomCalculated(index,zoomCalculate);
    }
}
void VDisplay::handleZoomTargetChanged(float zoomTarget){
    if(m_gimbal!= nullptr){
        m_gimbal->setZoomTarget(m_gimbal->context()->m_sensorID,zoomTarget);
    }
}
void VDisplay::init()
{
    // Using for yolov3-tiny_3l
    std::string names_file   = "../GPUBased/OD/yolo-setup/visdrone2019.names";
    std::string cfg_file     = "../GPUBased/OD/yolo-setup/yolov3-tiny_3l.cfg";
    std::string weights_file = "../GPUBased/OD/yolo-setup/yolov3-tiny_3l_last.weights";
    /*
     * @Editor: Giapvn
     * For testing of object detector with yolov3-tiny with coco
     */

//    std::string names_file   = "../GPUBased/OD/yolo-setup/coco.names";
//    std::string cfg_file     = "../GPUBased/OD/yolo-setup/yolov3-tiny.cfg";
//    std::string weights_file = "../GPUBased/OD/yolo-setup/yolov3-tiny.weights";

    std::string plate_cfg_file_click = "../GPUBased/Clicktrack/yolo-setup/yolov3-tiny_512.cfg";
    std::string plate_weights_file_click = "../GPUBased/Clicktrack/yolo-setup/yolov3-tiny_best.weights";
    std::string plate_cfg_search = "../GPUBased/plateOCR/yolo-setup/yolov3-tiny.cfg";
    std::string plate_weights_search = "../GPUBased/plateOCR/yolo-setup/yolov3-tiny_best.weights";
    m_detector = new Detector(cfg_file, weights_file);
    m_vODWorker->setDetector(m_detector);
    m_clicktrackDetector = new Detector(plate_cfg_file_click, plate_weights_file_click);
    m_vTrackWorker->setClicktrackDetector(m_clicktrackDetector);
    m_vTrackWorker->setObjDetector(m_detector);
    m_searchDetector = new Detector(plate_cfg_search, plate_weights_search);
    m_vSearchWorker->setPlateDetector(m_searchDetector);

    m_OCR = new OCR();
    m_vTrackWorker->setOCR(m_OCR);
    m_vTrackWorker->setDetector(m_detector);
    m_vSearchWorker->setOCR(m_OCR);
}

void VDisplay::start()
{
    m_vFrameGrabber->start();
    m_vPreprocess->start();
    m_vODWorker->start();
    m_vMOTWorker->start();
    m_vSearchWorker->start();
    m_vTrackWorker->start();
    m_threadEODisplay->start();
}

void VDisplay::setVideo(QString _ip, int _port)
{
    printf("%s - %s\r\n",__func__,_ip.toStdString().c_str());
    if(_ip.contains("filesrc")){
        Q_EMIT sourceLinkChanged(true);
    }else{
        Q_EMIT sourceLinkChanged(false);
    }
    m_vFrameGrabber->setSource(_ip.toStdString(), _port);
}
void VDisplay::setSensorColor(QString colorMode){
    m_vTrackWorker->setSensorColor(colorMode);
}
void VDisplay::setObjectDetect(bool enable){
    m_vDisplayWorker->m_enOD = enable;
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
    m_vTrackWorker->setPowerLineDetect(m_enPD);
}
void VDisplay::setPowerLineDetectRect(QRect rect){
    m_vTrackWorker->setPowerLineDetectRect(rect);
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
        m_gimbal->setDigitalStab(true);
    }
    m_vTrackWorker->setClick(_px, _py, _w, _h);
}

void VDisplay::setStab(bool _en)
{
    m_vTrackWorker->m_stabEnable = _en;
}

void VDisplay::setRecord(bool _en)
{
    m_vDisplayWorker->setVideoSavingState(_en);
}
void VDisplay::setObjectSearch(bool enable){
    m_vTrackWorker->m_objectSearch = enable;
}
void VDisplay::setShare(bool enable)
{
    m_vDisplayWorker->m_enShare = enable;
    if(enable){
#ifdef USE_VIDEO_CPU
    setSourceRTSP("( appsrc name=othersrc ! avenc_mpeg4 bitrate=1500000 ! rtpmp4vpay config-interval=3 name=pay0 pt=96 )",
                  8554,m_sourceSize.width(),m_sourceSize.height());
#endif
#ifdef USE_VIDEO_GPU
//    setSourceRTSP("( appsrc name=othersrc ! nvh264enc bitrate=1500000 ! h264parse ! rtph264pay mtu=1400 name=pay0 pt=96 )",
//                  8554,m_sourceSize.width(),m_sourceSize.height());
    setSourceRTSP("( appsrc name=othersrc ! videoscale ! video/x-raw,width=1280,height=720 ! avenc_mpeg4 bitrate=2000000 ! rtpmp4vpay config-interval=3 name=pay0 pt=96 )",
                  8554,m_sourceSize.width(),m_sourceSize.height());
#endif
    }else{
        stopRTSP();
    }
}
void VDisplay::goToPosition(float percent){
    m_vFrameGrabber->goToPosition(percent);
}
void VDisplay::setSpeed(float speed){
    m_vFrameGrabber->setSpeed(speed);
}
qint64 VDisplay::getTime(QString type){
    if(type == "TOTAL"){
        return m_vFrameGrabber->getTotalTime();
    }else if(type == "CURRENT"){
        return m_vFrameGrabber->getPosCurrent();
    }
}

void VDisplay::pause(bool pause){
    m_vFrameGrabber->pause(pause);
    m_vTrackWorker->pause(pause);
}
void VDisplay::stop()
{
    printf("\nSTOP===============================================================");
    m_vFrameGrabber->stopPipeline();
    m_vSavingWorker->stopPipeline();
    m_vODWorker->stop();
    m_vSearchWorker->stop();
    m_vMOTWorker->stop();
    m_vTrackWorker->stop();
    m_vPreprocess->stop();
    // stop rtsp
    stopRTSP();
    std::this_thread::sleep_for(std::chrono::seconds(1));
}
void VDisplay::capture(bool writeTime, bool writeLocation){
    m_vDisplayWorker->capture(writeTime,writeLocation);
}
void VDisplay::changeTrackSize(int _val)
{
    m_vTrackWorker->changeTrackSize(_val);
}
