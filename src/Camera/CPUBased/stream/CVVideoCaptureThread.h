#ifndef CVVIDEOCAPTURETHREAD_H
#define CVVIDEOCAPTURETHREAD_H

#include "Camera/Buffer/RollBuffer.h"
#include "Camera/Cache/TrackObject.h"
#include "Camera/VideoEngine/VideoEngineInterface.h"
#include "CVVideoCapture.h"
#include "CVVideoProcess.h"
#include "CVRecord.h"
class CVVideoCaptureThread : public VideoEngine
{
        Q_OBJECT
        /*Edit for getting motion data*/
    public:
        explicit CVVideoCaptureThread(VideoEngine *parent = 0);
        ~CVVideoCaptureThread();

        PlateLog* plateLog() override{return m_process->m_plateLog;};
        void setPlateLog(PlateLog* plateLog) override{
            if(m_process != nullptr){
                m_process->m_plateLog = plateLog;
            }
        }
        void setGimbal(GimbalInterface* gimbal) override;
    public:
        Q_INVOKABLE void setObjectDetect(bool enable) override{}
        Q_INVOKABLE void setPowerLineDetect(bool enable) override{}
        Q_INVOKABLE void start() override;
        Q_INVOKABLE void play() override;
        Q_INVOKABLE void stop() override;
        Q_INVOKABLE void setVideo(QString _ip, int _port=0) override;
        Q_INVOKABLE void setStab(bool enable) override;
        Q_INVOKABLE void setShare(bool enable) override;
        Q_INVOKABLE void setTrackState(bool enable) override;
        Q_INVOKABLE void capture() override;
        Q_INVOKABLE void updateFOV(float irFOV, float eoFOV) override;
        Q_INVOKABLE void changeTrackSize(int newSize) override;
        Q_INVOKABLE bool getTrackEnable() override;
        Q_INVOKABLE void setStreamMount(QString _streamMount) override;
        Q_INVOKABLE void disableObjectDetect() override;
        Q_INVOKABLE void enableObjectDetect() override;
        Q_INVOKABLE void setDigitalStab(bool _en) override;
        Q_INVOKABLE void setTrackAt(int _id, double _px, double _py, double _w, double _h) override;
        Q_INVOKABLE void setRecord(bool _en) override;
        Q_INVOKABLE void pause(bool pause) override;
        Q_INVOKABLE void goToPosition(float percent) override;
        Q_INVOKABLE void setSpeed(float speed) override;
        Q_INVOKABLE qint64 getTime(QString type) override;
public Q_SLOTS:
        void doShowVideo() override;
public:
        QThread *m_captureThread;
        QThread *m_processThread;
        QThread *m_recordThread;
        CVVideoCapture *m_capture = nullptr;
        CVVideoProcess *m_process = nullptr;
        CVRecord *m_record = nullptr;
        QMutex *m_mutexCapture;
        QMutex *m_mutexProcess;
        //    auto startTotal
        bool m_captureStopped = false;
        bool m_processStopped = false;
        bool m_recordStopped = false;
        std::deque<std::pair<int, GstSample *>> m_imageQueue;
};

#endif // CVVIDEOCAPTURETHREAD_H
