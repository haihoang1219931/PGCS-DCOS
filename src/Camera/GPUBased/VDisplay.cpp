#include "VDisplay.h"


VDisplay::VDisplay(QObject *_parent) : QObject(_parent)
{
    m_id = -1;
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
    connect(m_vTrackWorker, SIGNAL(determinedTrackObjected(int, double, double, double, double, double, double, double, double)),
            this, SLOT(slDeterminedTrackObjected(int, double, double, double, double, double, double, double, double)));
    connect(m_vTrackWorker, SIGNAL(determinedPlateOnTracking(QString, QString)),
            this, SIGNAL(determinedPlateOnTracking(QString, QString)));
    connect(m_vTrackWorker, SIGNAL(objectLost()),
            this, SLOT(slObjectLost()));
    connect(m_vDisplayWorker,&VDisplayWorker::readyDrawOnViewerID,this,&VDisplay::drawOnViewerID);
    m_videoSurfaceSize.setWidth(-1);
    m_videoSurfaceSize.setHeight(-1);
    init();
}

void VDisplay::setVideo(QString _ip, int _port)
{
    m_vFrameGrabber->setSource(_ip.toStdString(), _port);
    m_vFrameGrabber->restartPipeline();
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

int VDisplay::addSubViewer(ImageItem *viewer){
    printf("%s[%d] %p\r\n",__func__,m_listSubViewer.size(),viewer);
    this->freezeMap()[m_listSubViewer.size()] = true;
    m_listSubViewer.append(viewer);
    return m_listSubViewer.size() -1;
}
void VDisplay::removeSubViewer(int viewerID){
     printf("%s[%d] %d\r\n",__func__,m_listSubViewer.size(),viewerID);
    if(viewerID >= 0 && viewerID < m_listSubViewer.size()){
        this->freezeMap().remove(viewerID);
        m_listSubViewer.removeAt(viewerID);
    }
}
void VDisplay::start()
{

    m_vFrameGrabber->start();
    m_vPreprocess->start();
    m_vODWorker->start();
    m_vMOTWorker->start();
    m_vSearchWorker->start();
    m_vRTSPServer->start();
    m_vSavingWorker->initPipeline();
    m_vSavingWorker->start();
    m_vTrackWorker->start();
    m_threadEODisplay->start();
}

int VDisplay::frameID()
{
    return m_id;
}

QAbstractVideoSurface *VDisplay::videoSurface()
{
    return m_videoSurface;
}

void VDisplay::setVideoSurface(QAbstractVideoSurface *_videoSurface)
{
    printf("setVideoSurface");

    if (m_videoSurface != _videoSurface) {
        m_videoSurface = _videoSurface;
        update();
    }
}
void VDisplay::update()
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
QSize VDisplay::sourceSize()
{
    return m_sourceSize;
}


void VDisplay::slObjectLost(){
    removeTrackObjectInfo(0);
    Q_EMIT objectLost();
}
void VDisplay::slDeterminedTrackObjected(int _id, double _px, double _py, double _oW, double _oH, double _w, double _h,
                                         double _pxStab,double _pyStab){
    updateTrackObjectInfo("Object","RECT",QVariant(QRect(
                                                       static_cast<int>(_pxStab),
                                                       static_cast<int>(_pyStab),
                                                       static_cast<int>(_oW),
                                                       static_cast<int>(_oH)))
                          );
    updateTrackObjectInfo("Object","LATITUDE",QVariant(20.975092+_px/1000000));
    updateTrackObjectInfo("Object","LONGIITUDE",QVariant(105.307680+_py/1000000));
    updateTrackObjectInfo("Object","SPEED",QVariant(_py));
    updateTrackObjectInfo("Object","ANGLE",QVariant(_px));
    Q_EMIT determinedTrackObjected(_id,_px,_py,_oW, _oH, _w, _h,_pxStab,_pyStab);
}
void VDisplay::onReceivedFrame(int _id, QVideoFrame frame)
{
    if(m_videoSurface!=nullptr){
        m_id = _id;
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
        m_id = m_vDisplayWorker->m_currID;
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
void VDisplay::updateVideoSurface(int width, int height){
    m_updateCount = 0;
    m_updateVideoSurface = true;
    m_videoSurfaceSize.setWidth(width);
    m_videoSurfaceSize.setHeight(height);
}
void VDisplay::setVideoSource(QString _ip, int _port)
{
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

void VDisplay::setTrackAt(int _id, double _px, double _py, double _w, double _h)
{
    m_enTrack = true;
    m_enSteer = false;
    m_vTrackWorker->hasNewTrack(_id, _px, _py, _w, _h, false, m_vDisplayWorker->getDigitalStab());
    int x = static_cast<int>(_px/_w*m_sourceSize.width());
    int y = static_cast<int>(_py/_h*m_sourceSize.height());
    printf("%s at (%dx%d)\r\n",__func__,x,y);
    removeTrackObjectInfo(0);
    TrackObjectInfo *object = new TrackObjectInfo(m_sourceSize,QRect(x-20,y-20,40,40),"Object",20.975092,105.307680,0,0,"Track");
    object->setIsSelected(true);
    addTrackObjectInfo(object);
}

void VDisplay::setVideoSavingState(bool _state)
{
    m_vDisplayWorker->setVideoSavingState(_state);
}

void VDisplay::enVisualLock()
{
    m_enSteer = true;
    m_enTrack = false;
    m_vTrackWorker->hasNewTrack(-1, 1920 / 2, 1080 / 2, 1920, 1080, true, m_vDisplayWorker->getDigitalStab());
}

void VDisplay::disVisualLock()
{
    m_enSteer = false;
    m_enTrack = false;
    m_vTrackWorker->hasNewMode();
    m_vTrackWorker->disSteer();
}

void VDisplay::setDigitalStab(bool _en)
{
    m_vDisplayWorker->setDigitalStab(_en);
}

void VDisplay::setGimbalRecorder(bool _en)
{
    m_vDisplayWorker->setVideoSavingState(_en);
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

void VDisplay::changeTrackSize(int _val)
{
    m_vTrackWorker->changeTrackSize(_val);
}
void VDisplay::drawOnViewerID(cv::Mat img, int viewerID){
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
