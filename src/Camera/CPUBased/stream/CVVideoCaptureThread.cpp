#include "CVVideoCaptureThread.h"


CVVideoCaptureThread::CVVideoCaptureThread(VideoEngineInterface *parent) : VideoEngineInterface(parent)
{
    char cmd[100];
    std::string day = Utils::get_day();
#ifdef __linux__
    //linux code goes here
    m_logFolder = QGuiApplication::applicationDirPath().toStdString() + "/flights";
    sprintf(cmd, "/bin/mkdir -p %s", m_logFolder.c_str());
#elif _WIN32
    // windows code goes here
    m_logFolder = QGuiApplication::applicationDirPath().toStdString() + "/flights";
    sprintf(cmd, "mkdir %s", m_logFolder.c_str());
#else
#endif
    printf("cmd = %s\r\n", cmd);
    QDir dir(QString::fromStdString(m_logFolder));
    if (!dir.exists()) {
        dir.mkpath(QString::fromStdString(m_logFolder));
    }
    //    system(cmd);
    QProcess process;
    process.start(QString::fromStdString(cmd));
    process.waitForFinished(-1); // will wait forever until finished
    std::string timestamp = Utils::get_time_stamp();
    m_logFile = m_logFolder + "/" + timestamp;
    m_capture = new CVVideoCapture(nullptr);
    m_process = new CVVideoProcess(nullptr);
    m_record = new CVRecord(nullptr);
    m_captureThread = new QThread(nullptr);
    m_recordThread = new QThread(nullptr);
    m_processThread = new QThread(nullptr);
    m_mutexCapture = new QMutex();
    m_mutexProcess = new QMutex();
    m_vRTSPServer = new VRTSPServer;
    m_vSavingWorker = new VSavingWorker;
    m_capture->moveToThread(m_captureThread);
    m_process->moveToThread(m_processThread);
    m_record->moveToThread(m_recordThread);
    connect(m_captureThread, SIGNAL(started()), m_capture, SLOT(doWork()));
    connect(m_processThread, SIGNAL(started()), m_process, SLOT(doWork()));
    connect(m_recordThread, SIGNAL(started()), m_record, SLOT(doWork()));
    connect(m_process, SIGNAL(trackInitSuccess(bool, int, int, int, int)), this, SIGNAL(trackInitSuccess(bool, int, int, int, int)));
    connect(m_process, SIGNAL(processDone()), this, SLOT(doShowVideo()));
    connect(m_process, SIGNAL(trackStateLost()), this, SLOT(slObjectLost()));
    connect(m_process, SIGNAL(trackStateFound(int, double, double, double, double, double, double)), this,
            SLOT(slDeterminedTrackObjected(int, double, double, double, double, double, double)));
    connect(this, SIGNAL(sourceSizeChanged(int, int)),
            this, SLOT(onStreamFrameSizeChanged(int, int)));
    connect(m_process,&CVVideoProcess::readyDrawOnViewerID,this,&CVVideoCaptureThread::drawOnViewerID);
    //    connect(m_process,SIGNAL(objectSizeChange(float)),this,SLOT(doChangeZoom(float)));
    m_capture->m_imageQueue = &m_imageQueue;
    m_capture->m_mutexCapture = m_mutexCapture;
    m_capture->m_logFolder = m_logFolder;
    m_capture->m_logFile = m_logFile;
    m_process->m_imageQueue = &m_imageQueue;
    m_process->m_frameID = &m_frameID;
    m_process->m_imgShow = &m_imgShow;
    m_process->m_mutexCapture = m_mutexCapture;
    m_process->m_mutexProcess = m_mutexProcess;
    m_process->m_logFolder = m_logFolder;
    m_process->m_logFile = m_logFile;
    m_record->m_imageQueue = &m_imageQueue;
    m_record->m_mutexCapture = m_mutexCapture;
    m_record->m_logFolder = m_logFolder;
    m_record->m_logFile = m_logFile;
    m_gstRTSPBuff = new RollBuffer_<GstFrameCacheItem>(10);
    m_buffVideoSaving = new RollBuffer_<GstFrameCacheItem>(10);
    m_process->m_gstRTSPBuff = m_gstRTSPBuff;
    m_process->m_buffVideoSaving = m_buffVideoSaving;
    m_vSavingWorker->m_buffVideoSaving = m_buffVideoSaving;
    m_vRTSPServer->m_gstRTSPBuff = m_gstRTSPBuff;
}
CVVideoCaptureThread::~CVVideoCaptureThread()
{
    printf("Destroy CVVideoCaptureThread\r\n");
    stop();
    m_captureThread->wait(100);
    m_captureThread->quit();

    if (!m_captureThread->wait(100)) {
        m_captureThread->terminate();
        m_captureThread->wait(100);
    }

    printf("Capture thread stopped\r\n");
    m_processThread->wait(100);
    m_processThread->quit();

    if (!m_processThread->wait(100)) {
        m_processThread->terminate();
        m_processThread->wait(100);
    }

    printf("Process thread stopped\r\n");
    m_recordThread->wait(100);
    m_recordThread->quit();

    if (!m_recordThread->wait(100)) {
        m_recordThread->terminate();
        m_recordThread->wait(100);
    }

    printf("Record thread stopped\r\n");
    m_vRTSPServer->wait(100);
    m_vRTSPServer->quit();

    if (!m_vRTSPServer->wait(100)) {
        m_vRTSPServer->terminate();
        m_vRTSPServer->wait(100);
    }

    printf("RTSP thread stopped\r\n");

    m_vSavingWorker->wait(100);
    m_vSavingWorker->quit();

    if (!m_vSavingWorker->wait(100)) {
        m_vSavingWorker->terminate();
        m_vSavingWorker->wait(100);
    }

    printf("Saving thread stopped\r\n");

    m_vSavingWorker->deleteLater();
    m_vRTSPServer->deleteLater();
    m_recordThread->deleteLater();
    m_record->deleteLater();
    m_processThread->deleteLater();
    m_process->deleteLater();
    m_captureThread->deleteLater();
    m_capture->deleteLater();
    delete m_mutexCapture;
    delete m_mutexProcess;
}
void CVVideoCaptureThread::setVideo(QString _ip, int _port)
{
    m_capture->m_ip = _ip.toStdString();
    m_capture->m_port = _port;
    m_capture->setSource(_ip.toStdString() + " ! appsink name=mysink sync=true async=true");
}
void CVVideoCaptureThread::start()
{
    //    m_captureThread->wait(100);
    m_captureThread->start();
    //    m_processThread->wait(100);
    m_processThread->start();
    //    m_recordThread->wait(100);
    //    m_recordThread->start();
}
void CVVideoCaptureThread::play()
{
    //    m_captureThread->wait(100);
    m_captureThread->start();
    //    m_processThread->wait(100);
    m_processThread->start();
    //    m_recordThread->wait(100);
    //    m_recordThread->start();
}
void CVVideoCaptureThread::stop()
{
    printf("STOP===============================================================STOP\r\n");

    if (m_capture->m_stop == false) {
        m_capture->setStateRun(false);
    } else {
        printf("Capture already stopped\r\n");
    }

    if (m_process->m_stop == false) {
        m_process->m_stop = true;
    } else {
        printf("Process already stopped\r\n");
    }

    if (m_record->m_stop == false) {
        m_record->m_stop = true;
    } else {
        printf("Record already stopped\r\n");
    }
    m_vRTSPServer->setStateRun(false);
    m_vSavingWorker->stopPipeline();
}

void CVVideoCaptureThread::setStab(bool enable)
{
    m_process->m_stabEnable = enable;
}
void CVVideoCaptureThread::setRecord(bool enable)
{
    m_process->m_recordEnable = enable;
}
void CVVideoCaptureThread::setShare(bool enable)
{
    m_process->m_sharedEnable = enable;
}
void CVVideoCaptureThread::setTrackState(bool enable)
{
    m_process->m_trackEnable = enable;
}
void CVVideoCaptureThread::capture()
{
    m_process->capture();
}
void CVVideoCaptureThread::updateFOV(float eoFOV, float irFOV)
{
    m_process->m_eoFOV = eoFOV;
    m_process->m_irFOV = irFOV;
}
void CVVideoCaptureThread::stopTrack(bool enable)
{
    m_process->m_trackEnable = enable;

    if (m_process->m_tracker->isInitialized()) {
        while (m_process->m_tracker->isRunning());

        m_process->m_tracker->resetTrack();
    }

    if (m_process->k_tracker->isInititalized()) {
        while (m_process->k_tracker->isRunning());

        m_process->k_tracker->resetTrack();
    }

    if (m_process->thresh_tracker->isInitialized()) {
        while (m_process->thresh_tracker->isRunning());

        m_process->thresh_tracker->resetTrack();
    }

    m_process->object_position = cv::Rect(0, 0, 0, 0);
}
bool CVVideoCaptureThread::getTrackEnable()
{
    return (m_process->m_tracker->isInitialized() || m_process->m_tracker->isRunning());
}
void CVVideoCaptureThread::changeTrackSize(int newSize)
{
    m_process->m_trackSize = newSize;
}
void CVVideoCaptureThread::doShowVideo()
{
    if (m_videoSurface != NULL) {
        //        m_mutexCapture->lock();
        if (m_sourceSize.width() != m_imgShow.cols ||
            m_sourceSize.height() != m_imgShow.rows) {
            m_sourceSize.setWidth(m_imgShow.cols);
            m_sourceSize.setHeight(m_imgShow.rows);
            Q_EMIT sourceSizeChanged(m_imgShow.cols, m_imgShow.rows);
        }
        if(m_updateVideoSurface){
            update();
            m_updateVideoSurface = false;
        }
        QImage tmp((uchar *)m_imgShow.data, m_imgShow.cols, m_imgShow.rows, QImage::Format_RGBA8888);
        QVideoFrame output = QVideoFrame(tmp);

//        printf("show image[%dx%d]\r\n",m_imgShow.cols,m_imgShow.rows);
        if (!m_videoSurface->present(output)) {
//            printf("Show failed\r\n");
        } else {
//            printf("Show success\r\n");
        }
    }
}

void CVVideoCaptureThread::disableObjectDetect(){

}
void CVVideoCaptureThread::enableObjectDetect(){

}
void CVVideoCaptureThread::enVisualLock(){

}
void CVVideoCaptureThread::disVisualLock(){

}
void CVVideoCaptureThread::setDigitalStab(bool _en){
    m_process->m_stabEnable = _en;
}
void CVVideoCaptureThread::setTrackAt(int _id, double _px, double _py, double _w, double _h)
{
    int x = static_cast<int>(_px/_w*m_sourceSize.width());
    int y = static_cast<int>(_py/_h*m_sourceSize.height());
    m_process->setTrack(x,y);
    printf("%s at (%dx%d)\r\n",__func__,x,y);
    removeTrackObjectInfo(0);
    TrackObjectInfo *object = new TrackObjectInfo(m_sourceSize,QRect(x-20,y-20,40,40),"Object",20.975092,105.307680,0,0,"Track");
    object->setIsSelected(true);
    addTrackObjectInfo(object);
}
void CVVideoCaptureThread::setStreamMount(QString _streamMount)
{
    m_vRTSPServer->setStreamMount(_streamMount.toStdString());
}
void CVVideoCaptureThread::onStreamFrameSizeChanged(int width, int height)
{
    printf("%s [%dx%d]\r\n",__func__,width,height);
    m_vRTSPServer->setStreamSize(width, height);
    m_vSavingWorker->setStreamSize(width, height);
//    m_vSavingWorker->setSensorMode(m_sensorMode);

    if (m_enStream) {
        m_vRTSPServer->start();
    }

    if (m_enSaving) {
        m_vSavingWorker->start();
    }
}
