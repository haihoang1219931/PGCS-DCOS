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
#include "VSavingWorker.h"
#include "VRTSPServer.h"
#include <QVariantList>
#include <QVariant>
#include "OD/yolo_v2_class.hpp"
#include "Clicktrack/recognition.h"
#include "../../Camera/Cache/TrackObject.h"

class VDisplay : public QObject
{
        Q_OBJECT
        Q_PROPERTY(QQmlListProperty<TrackObjectInfo> listTrackObjectInfos READ listTrackObjectInfos NOTIFY listTrackObjectInfosChanged);
        Q_PROPERTY(QAbstractVideoSurface *videoSurface READ videoSurface WRITE
                   setVideoSurface)
        Q_PROPERTY(QSize sourceSize READ sourceSize NOTIFY sourceSizeChanged)
        Q_PROPERTY(int frameID READ frameID)
        Q_PROPERTY(bool enOD READ enOD)
        Q_PROPERTY(bool enTrack READ enTrack)
        Q_PROPERTY(bool enSteer READ enSteer)
        Q_PROPERTY(PlateLog* plateLog READ plateLog WRITE setPlateLog NOTIFY plateLogChanged)
    public:
        explicit VDisplay(QObject *_parent = 0);
        ~VDisplay();
    PlateLog* plateLog(){return m_vDisplayWorker->m_plateLog;};
    void setPlateLog(PlateLog* plateLog){
        if(m_vDisplayWorker != nullptr)
            m_vDisplayWorker->m_plateLog = plateLog;
            m_vTrackWorker->m_plateLog = plateLog;
        };
        QAbstractVideoSurface *videoSurface();
        void setVideoSurface(QAbstractVideoSurface *_videoSurface);
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
        QSize sourceSize();
        int frameID();
        void init();
        void stop();
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
        void update();
    Q_SIGNALS:
        void listTrackObjectInfosChanged();
        void determinedTrackObjected(int _id, double _px, double _py, double _oW, double _oH, double _w, double _h,
                                     double _pxStab,double _pyStab);
        void objectLost();
        void determinedPlateOnTracking(QString _imgPath, QString _plateID);
        void plateLogChanged();
    public Q_SLOTS:

        void onReceivedFrame(int _id, QVideoFrame _frame);
        void onReceivedFrame();
        void slDeterminedTrackObjected(int _id, double _px, double _py, double _oW, double _oH, double _w, double _h,
                                       double _pxStab,double _pyStab);
        void slObjectLost();
    public:
        Q_INVOKABLE void start();
        Q_INVOKABLE void updateVideoSurface();
        Q_INVOKABLE void setObjectDetect(bool enable);
        Q_INVOKABLE void setPowerLineDetect(bool enable);
        Q_INVOKABLE void setVideoSource(QString _ip, int _port);
        Q_INVOKABLE void searchByClass(QVariantList _classList);
        Q_INVOKABLE void setTrackAt(int _id, double _px, double _py, double _w, double _h);
        Q_INVOKABLE void disableObjectDetect();
        Q_INVOKABLE void enableObjectDetect();
        Q_INVOKABLE void setVideoSavingState(bool _state);
        Q_INVOKABLE void setVideo(QString _ip, int _port = 0);
        Q_INVOKABLE void enVisualLock();
        Q_INVOKABLE void disVisualLock();
        Q_INVOKABLE void setDigitalStab(bool _en);
        Q_INVOKABLE void setGimbalRecorder(bool _en);
        Q_INVOKABLE void changeTrackSize(int _val);

    Q_SIGNALS:
        void sourceSizeChanged(int newWidth, int newHeight);

    private:
        const QVideoFrame::PixelFormat VIDEO_OUTPUT_FORMAT =
            QVideoFrame::PixelFormat::Format_RGB32;
        QAbstractVideoSurface *m_videoSurface = nullptr;
        QSize m_sourceSize;
        QThread *m_threadEODisplay;
        VDisplayWorker *m_vDisplayWorker;
        VFrameGrabber *m_vFrameGrabber;
        VPreprocess *m_vPreprocess;
        VODWorker *m_vODWorker;
        VMOTWorker *m_vMOTWorker;
		VSearchWorker *m_vSearchWorker;
        VTrackWorker *m_vTrackWorker;
        VSavingWorker *m_vSavingWorker;
        VRTSPServer *m_vRTSPServer;
        bool m_updateVideoSurface = false;
        int m_id;

        // OD
        Detector *m_detector;
        bool m_enSteer = false;
        bool m_enTrack = false;
        bool m_enOD = false;
        bool m_enPD = false;

        // CLicktrack
        Detector *m_clicktrackDetector;
        OCR *m_OCR;
		// Search
        Detector *m_searchDetector;
private:
        QList<TrackObjectInfo *> m_listTrackObjectInfos;
};

#endif // VDISPLAY_H
