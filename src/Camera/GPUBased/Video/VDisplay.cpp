#include "VDisplay.h"


VDisplay::VDisplay(QObject *_parent) : QObject(_parent)
{
    m_id = -1;
    m_vFrameGrabber = new VFrameGrabber;
    m_vPreprocess = new VPreprocess;
    m_vODWorker = new VODWorker;
    m_vTrackWorker = new VTrackWorker;
    m_vRTSPServer = new VRTSPServer;
    m_vSavingWorker = new VSavingWorker("EO");
    m_threadDisplay = new QThread(0);
    m_vDisplayWorker = new VDisplayWorker(0);
    m_vDisplayWorker->moveToThread(m_threadDisplay);
    connect(m_vDisplayWorker, SIGNAL(receivedFrame(int, QVideoFrame)), this,
            SLOT(onReceivedFrame(int, QVideoFrame)));
    connect(m_vDisplayWorker, SIGNAL(receivedFrame()), this,
            SLOT(onReceivedFrame()));
    connect(m_threadDisplay, SIGNAL(started()), m_vDisplayWorker,
            SLOT(process()));
    connect(m_vTrackWorker, SIGNAL(determinedTrackObjected(int, double, double, double, double, double, double)),
            this, SLOT(onDeterminedTrackObjected(int, double, double, double, double, double, double)));
    connect(m_vTrackWorker, SIGNAL(determinedPlateOnTracking(QString, QString)),
            this, SLOT(onDeterminedPlateOnTracking(QString, QString)));
    connect(m_vFrameGrabber, SIGNAL(pipelineError(int)),
            this, SLOT(onPipelineError(int)));
    init();
}

VDisplay::~VDisplay() {}

void VDisplay::init()
{
    std::string names_file = "../src/Camera/GPUBased/Video/Multitracker/vehicle-weight/visdrone2019.names";
    std::string cfg_file = "../src/Camera/GPUBased/Video/Multitracker/vehicle-weight/yolov3-tiny_3l.cfg";
    std::string weights_file = "../src/Camera/GPUBased/Video/Multitracker/vehicle-weight/yolov3-tiny_3l_last.weights";
    std::string plate_cfg_file = "../src/Camera/GPUBased/Video/Multitracker/plate-weight/yolov3-tiny.cfg";
    std::string plate_weights_file = "../src/Camera/GPUBased/Video/Multitracker/plate-weight/yolov3-tiny_best.weights";
    m_mulTracker = new MultiTrack(cfg_file, weights_file);
    m_plateOCR = new PlateOCR(plate_cfg_file, plate_weights_file);
//    m_mulTracker->setPlateOCR(m_plateOCR);
    m_vODWorker->setMultiTracker(m_mulTracker);
	m_vODWorker->setPlateOCR(m_plateOCR);
    m_vTrackWorker->setPlateOCR(m_plateOCR);
}


void VDisplay::play()
{
    m_vFrameGrabber->initPipeline();
    m_vFrameGrabber->start();
    m_vPreprocess->start();
    m_vODWorker->start();
    m_vTrackWorker->start();
    m_threadDisplay->start();
    m_vRTSPServer->start();
    m_vSavingWorker->initPipeline();
    m_vSavingWorker->start();
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

    if (m_videoSurface != _videoSurface && m_videoSurface &&
        m_videoSurface->isActive()) {
        m_videoSurface->stop();
    }

    m_videoSurface = _videoSurface;

    if (m_videoSurface) {
        if (!m_videoSurface->start(
                QVideoSurfaceFormat(QSize(), VIDEO_OUTPUT_FORMAT))) {
            printf("Could not start QAbstractVideoSurface, error: %d",
                   m_videoSurface->error());
        } else {
            printf("Start QAbstractVideoSurface done\r\n");
        }
    }
}

QSize VDisplay::sourceSize()
{
    return m_sourceSize;
}

void VDisplay::onReceivedFrame(int _id, QVideoFrame _frame)
{
    m_id = _id;
    m_videoSurface->present(_frame);
}

void VDisplay::onReceivedFrame()
{
    m_id = m_vDisplayWorker->m_currID;
    QVideoFrame frame = QVideoFrame(
                            QImage((uchar *)m_vDisplayWorker->m_imgShow.data,
                                   m_vDisplayWorker->m_imgShow.cols,
                                   m_vDisplayWorker->m_imgShow.rows, QImage::Format_RGBA8888));
    frame.map(QAbstractVideoBuffer::ReadOnly);
    m_videoSurface->present(frame);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    frame.unmap();
}

void VDisplay::onDeterminedTrackObjected(int _id, double _px, double _py, double _w, double _h, double _oW, double _oH)
{
    //    printf("\n---> %d - %0.3f - %0.3f - %0.3f - %0.3f - %0.3f - %0.3f ", _id, _px, _py, _w, _h, _oW, _oH);
    Q_EMIT determinedTrackObjected(_id, _px, _py, _w, _h, _oW, _oH);
}

void VDisplay::onDeterminedPlateOnTracking(QString _imgPath, QString _plateID)
{
    printf("\n---> Emit determinedPlateOnTracking (%s, %s) ", _imgPath.toStdString().data(), _plateID.toStdString().data());
    Q_EMIT determinedPlateOnTracking(_imgPath, _plateID);
}

void VDisplay::setVideoSource(QString _ip, int _port)
{
    m_vFrameGrabber->setSource(_ip.toStdString(), _port);
}

void VDisplay::searchByClass(QVariantList _classList)
{
    std::vector<int> classIDs;

    if (_classList.size() == 0) {
        m_vODWorker->disableOD();
    }

    if (m_vTrackWorker->isRunning()) {
        m_vTrackWorker->hasNewMode();
    }

    for (int i = 0; i < _classList.size(); i++) {
        classIDs.push_back(_classList.at(i).toInt());
    }

    m_vDisplayWorker->setListObjClassID(classIDs);
    m_vODWorker->enableOD();
}

void VDisplay::disableObjectDetect()
{
    m_vODWorker->disableOD();
}

void VDisplay::enableObjectDetect()
{
    std::vector<int> classIDs;

    for (int i = 0; i < 12; i++) {
        classIDs.push_back(i);
    }

    if (m_vTrackWorker->isRunning()) {
        m_vTrackWorker->hasNewMode();
    }

    m_vDisplayWorker->setListObjClassID(classIDs);
    m_vODWorker->enableOD();
}

void VDisplay::setVideoSavingState(bool _state)
{
    m_vDisplayWorker->setVideoSavingState(_state);
}

void VDisplay::setTrackAt(int _id, double _px, double _py, double _w,
                          double _h)
{
    if (m_vODWorker->isRunning()) {
        m_vODWorker->disableOD();
    }

    m_vTrackWorker->hasNewTrack(_id, _px, _py, _w, _h);
}

void VDisplay::onPipelineError(int _codeError)
{
    //FrameGrabber need to do smt
}
