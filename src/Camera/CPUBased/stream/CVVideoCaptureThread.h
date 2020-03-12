#ifndef CVVIDEOCAPTURETHREAD_H
#define CVVIDEOCAPTURETHREAD_H

#include <QObject>
#include <QThread>
#include <QProcess>
#include <QDir>
#include <QAbstractVideoSurface>
#include <QVideoSurfaceFormat>
#include <QMetaType>
#include <QQmlListProperty>
#include <QThread>
#include <QVariantList>
#include <QVariantMap>
#include <QtQuick>
#include <QQmlApplicationEngine>
#include <QQmlListProperty>
#include <QList>
#include <QRect>
#include <chrono>
#include "../../ControllerLib/Buffer/RollBuffer.h"
#include "../../Cache/TrackObject.h"
#include "CVVideoCapture.h"
#include "CVVideoProcess.h"
#include "CVRecord.h"
#include "VRTSPServer.h"
#include "VSavingWorker.h"

class CVVideoCaptureThread : public QObject
{
        Q_OBJECT
        Q_PROPERTY(QQmlListProperty<TrackObjectInfo> listTrackObjectInfos READ listTrackObjectInfos NOTIFY listTrackObjectInfosChanged);
        Q_PROPERTY(QAbstractVideoSurface *videoSurface READ videoSurface WRITE setVideoSurface)
        Q_PROPERTY(QSize sourceSize READ sourceSize NOTIFY sourceSizeChanged)
        Q_PROPERTY(QString sensorTrack READ sensorTrack WRITE setSensorTrack)
        Q_PROPERTY(bool enStream READ enStream WRITE setEnStream)
        Q_PROPERTY(bool enSaving READ enSaving WRITE setEnSaving)
        Q_PROPERTY(int sensorMode READ sensorMode WRITE setSensorMode)
        Q_PROPERTY(PlateLog* plateLog READ plateLog WRITE setPlateLog NOTIFY plateLogChanged)
        Q_PROPERTY(int frameID READ frameID)
        Q_PROPERTY(bool enOD READ enOD)
        Q_PROPERTY(bool enTrack READ enTrack)
        Q_PROPERTY(bool enSteer READ enSteer)
        /*Edit for getting motion data*/
    public:
        explicit CVVideoCaptureThread(QObject *parent = 0);
        virtual ~CVVideoCaptureThread();
        QQmlListProperty<TrackObjectInfo> listTrackObjectInfos()
        {
            return QQmlListProperty<TrackObjectInfo>(this, m_listTrackObjectInfos);
        }
        void addTrackObjectInfo(TrackObjectInfo* object)
        {
            this->m_listTrackObjectInfos.append(object);
            Q_EMIT listTrackObjectInfosChanged();
        }
        void removeTrackObjectInfo(const int& sequence) {
            if(sequence < 0 || sequence >= this->m_listTrackObjectInfos.size()){
                return;
            }

            // remove user on list
            this->m_listTrackObjectInfos.removeAt(sequence);
            Q_EMIT listTrackObjectInfosChanged();
        }
        void removeTrackObjectInfo(const QString &userUid)
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
        Q_INVOKABLE void updateTrackObjectInfo(const QString& userUid, const QString& attr, const QVariant& newValue) {

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
        QString sensorTrack()
        {
            return m_process->m_sensorTrack;
        }
        void setSensorTrack(QString sensorTrack)
        {
            m_process->m_sensorTrack = sensorTrack;
        }
        void setEnStream(bool _enStream)
        {
            m_enStream = _enStream;
        }
        bool enStream()
        {
            return m_enStream;
        }
        void setEnSaving(bool _enSaving)
        {
            m_enSaving = _enSaving;
        }
        bool enSaving()
        {
            return m_enSaving;
        }
        void setSensorMode(bool _sensorMode)
        {
            m_sensorMode = _sensorMode;
        }
        bool sensorMode()
        {
            return m_sensorMode;
        }
        int frameID(){
            return m_id;
        }
        bool enOD()
        {
            return m_enOD;
        }
        bool enTrack()
        {
            return m_enTrack;
        }
        bool enSteer()
        {
            return m_enSteer;
        }

        PlateLog* plateLog(){return m_process->m_plateLog;};
        void setPlateLog(PlateLog* plateLog){
            if(m_process != nullptr){
                m_process->m_plateLog = plateLog;
            }
        }

    public:
        Q_INVOKABLE void start();
        Q_INVOKABLE void play();
        Q_INVOKABLE void stop();
        Q_INVOKABLE void setVideo(QString _streamingAddress);
        Q_INVOKABLE void setAddress(QString _ip, int _port);
        Q_INVOKABLE void setTrack(int x, int y);
        Q_INVOKABLE void setStab(bool enable);
        Q_INVOKABLE void setRecord(bool enable);
        Q_INVOKABLE void setShare(bool enable);
        Q_INVOKABLE void setTrackState(bool enable);
        Q_INVOKABLE void capture();
        Q_INVOKABLE void updateFOV(float irFOV, float eoFOV);
        Q_INVOKABLE void stopTrack(bool enable);
        Q_INVOKABLE void changeTrackSize(int newSize);
        Q_INVOKABLE bool getTrackEnable();
        Q_INVOKABLE void setStreamMount(QString _streamMount);
        Q_INVOKABLE void disableObjectDetect();
        Q_INVOKABLE void enableObjectDetect();
        Q_INVOKABLE void enVisualLock();
        Q_INVOKABLE void disVisualLock();
        Q_INVOKABLE void setDigitalStab(bool _en);
        Q_INVOKABLE void setTrackAt(int _id, double _px, double _py, double _w, double _h);
    public:
        QAbstractVideoSurface *videoSurface();
        void setVideoSurface(QAbstractVideoSurface *videoSurface);
        QSize sourceSize();
        void update();
        bool allThreadStopped();
    Q_SIGNALS:
        void listTrackObjectInfosChanged();
        void sourceSizeChanged(int newWidth, int newHeight);
        void readyToRead();
        void determinedTrackObjected(int _id, double _px, double _py, double _oW, double _oH, double _w, double _h);
        void objectLost();
        void needZoomChange(float deltaFOV);
        void started();
        void stopped();
        void objectDetected();
        void trackInitSuccess(bool success, int _x, int _y, int _width, int _height);
        void plateLogChanged();
public Q_SLOTS:
        void setTrackType(QString trackType)
        {
            m_process->setTrackType(trackType);
        }
        void slDeterminedTrackObjected(int _id, double _px, double _py, double _oW, double _oH, double _w, double _h);
        void slObjectLost();
        void onStreamFrameSizeChanged(int width, int height);
        void doShowVideo();
        void restart();
        void killCaptureThread();
        void killProcessThread();
        void killRecordThread();
        void objectDetect();
        void enableDetect(bool enable);
        void setDetectSize(int size);
        void setObjectArea(int area);
        void doChangeZoom(float zoomRate);
public:
        const QVideoFrame::PixelFormat VIDEO_OUTPUT_FORMAT = QVideoFrame::PixelFormat::Format_RGB32;
        QThread *m_captureThread;
        QThread *m_processThread;
        QThread *m_recordThread;
        CVVideoCapture *m_capture = NULL;
        CVVideoProcess *m_process = NULL;
        CVRecord *m_record = NULL;
        QMutex *m_mutexCapture;
        QMutex *m_mutexProcess;
        //    auto startTotal;
        bool m_captureStopped = false;
        bool m_processStopped = false;
        bool m_recordStopped = false;
        std::deque<std::pair<int, GstSample *>> m_imageQueue;
        int m_frameID;
        cv::Mat m_imgShow;
        QSize m_sourceSize;
        QAbstractVideoSurface *m_videoSurface = NULL;
        std::string m_logFolder;
        std::string m_logFile;
        VRTSPServer *m_vRTSPServer;
        VSavingWorker *m_vSavingWorker;
        RollBuffer_<GstFrameCacheItem> *m_gstRTSPBuff;
        RollBuffer_<GstFrameCacheItem> *m_buffVideoSaving;
        bool m_enStream = true;
        bool m_enSaving = false;
        int m_sensorMode = -1;
        int m_id;

        // OD
        bool m_enSteer = false;
        bool m_enTrack = false;
        bool m_enOD = false;
private:
        QList<TrackObjectInfo *> m_listTrackObjectInfos;
};

#endif // CVVIDEOCAPTURETHREAD_H
