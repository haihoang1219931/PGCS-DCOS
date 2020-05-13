#ifndef VDISPLAY_H
#define VDISPLAY_H

#include "VDisplayWorker.h"
#include <QAbstractVideoSurface>
#include <QThread>
#include <QVideoSurfaceFormat>
#include <opencv2/opencv.hpp>
#include "VFrameGrabber.h"
#include "VPreprocess.h"
#include "VTrackWorker.h"
#include "VODWorker.h"
#include "VMOTWorker.h"
#include "VSearchWorker.h"
#include "Camera/VideoEngine/VSavingWorker.h"
#include "Camera/VideoEngine/VRTSPServer.h"
#include <QVariantList>
#include <QVariant>
#include "OD/yolo_v2_class.hpp"
#include "Clicktrack/recognition.h"
#include "Camera/VideoEngine/VideoEngineInterface.h"
#include "Camera/VideoDisplay/ImageItem.h"
#include "Camera/Cache/TrackObject.h"
class VDisplay : public VideoEngine
{
    Q_OBJECT
public:
    explicit VDisplay(VideoEngine *_parent = 0);
    ~VDisplay();
    PlateLog* plateLog() override{return m_vDisplayWorker->m_plateLog;};
    void setPlateLog(PlateLog* plateLog) override{
        if(m_vDisplayWorker != nullptr)
            m_vDisplayWorker->m_plateLog = plateLog;
        m_vTrackWorker->m_plateLog = plateLog;
    }
    void init();
    void setGimbal(GimbalInterface* gimbal) override;   
public Q_SLOTS:
    void onReceivedFrame(int _id, QVideoFrame _frame);
    void onReceivedFrame();
    void handleZoomTargetChangeStopped(float zoomTarget);
    void handleZoomCalculateChanged(int index,float zoomCalculate);
    void handleZoomTargetChanged(float zoomTarget);
public:
    Q_INVOKABLE void moveImage(float panRate,float tiltRate,float zoomRate,float alpha = 0) override;
    Q_INVOKABLE void start() override;
    Q_INVOKABLE void stop() override;
    Q_INVOKABLE void capture() override;
    Q_INVOKABLE void setObjectDetect(bool enable) override;
    Q_INVOKABLE void setPowerLineDetect(bool enable) override;
    Q_INVOKABLE void searchByClass(QVariantList _classList);
    Q_INVOKABLE void setTrackAt(int _id, double _px, double _py, double _w, double _h) override;
    Q_INVOKABLE void disableObjectDetect() override;
    Q_INVOKABLE void enableObjectDetect() override;
    Q_INVOKABLE void setVideoSavingState(bool _state);
    Q_INVOKABLE void setVideo(QString _ip, int _port = 0) override;
    Q_INVOKABLE void setStab(bool _en) override;
    Q_INVOKABLE void setRecord(bool _en) override;
    Q_INVOKABLE void changeTrackSize(int _val) override;
    Q_INVOKABLE void setShare(bool enable) override;
    Q_INVOKABLE void pause(bool pause) override;
    Q_INVOKABLE void goToPosition(float percent) override;
    Q_INVOKABLE void setSpeed(float speed) override;
    Q_INVOKABLE qint64 getTime(QString type) override;
private:
    QThread *m_threadEODisplay;
    VDisplayWorker *m_vDisplayWorker;
    VFrameGrabber *m_vFrameGrabber;
    VPreprocess *m_vPreprocess;
    VODWorker *m_vODWorker;
    VMOTWorker *m_vMOTWorker;
    VSearchWorker *m_vSearchWorker;
    VTrackWorker *m_vTrackWorker;
    // OD
    Detector *m_detector;
    // CLicktrack
    Detector *m_clicktrackDetector;
    OCR *m_OCR;
    // Search
    Detector *m_searchDetector;
};

#endif // VDISPLAY_H
