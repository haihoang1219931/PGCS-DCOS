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
    connect(m_vTrackWorker, SIGNAL(determinedTrackObjected(int, double, double, double, double, double, double)),
            this, SIGNAL(determinedTrackObjected(int, double, double, double, double, double, double)));
    connect(m_vTrackWorker, SIGNAL(determinedPlateOnTracking(QString, QString)),
            this, SIGNAL(determinedPlateOnTracking(QString, QString)));
    connect(m_vTrackWorker, SIGNAL(objectLost()),
            this, SIGNAL(objectLost()));
    init();
}

void VDisplay::resetVideoSource(QString _ip, int _port)
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
    std::string names_file   = "../Controller/Video/OD/yolo-setup/visdrone2019.names";
    std::string cfg_file     = "../Controller/Video/OD/yolo-setup/yolov3-tiny_3l.cfg";
    std::string weights_file = "../Controller/Video/OD/yolo-setup/yolov3-tiny_3l_last.weights";
	std::string plate_cfg_file_click = "../Controller/Video/Clicktrack/yolo-setup/yolov3-tiny_512.cfg";
	std::string plate_weights_file_click = "../Controller/Video/Clicktrack/yolo-setup/yolov3-tiny_best.weights";
	std::string plate_cfg_search = "../Controller/Video/plateOCR/yolo-setup/yolov3-tiny.cfg";
    std::string plate_weights_search = "../Controller/Video/plateOCR/yolo-setup/yolov3-tiny_best.weights";
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


void VDisplay::play()
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
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    frame.unmap();
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
