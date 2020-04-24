/**
 *========================================================================
 * Project: %{ProjectName}
 * Module: VideoEngineInterface.h
 * Module Short Description:
 * Author: %{AuthorName}
 * Date: %{Date}
 * Organization: Viettel Aerospace Institude - Viettel Group
 * =======================================================================
 */


#ifndef VIDEOENGINEINTERFACE_H
#define VIDEOENGINEINTERFACE_H

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
#include <QList>
#include <QtQuick>
#include <QQmlApplicationEngine>
#include <QQmlListProperty>
#include <QList>
#include <QRect>
#include "../ControllerLib/Buffer/RollBuffer.h"
#include "../ControllerLib/Buffer/RollBuffer_.h"
#include "../Cache/TrackObject.h"
#include "../Cache/GstFrameCacheItem.h"
#include "../../../Files/PlateLog.h"
#include <opencv2/core.hpp>
class VRTSPServer;
class VSavingWorker;
class ImageItem;
class TrackObjectInfo;
class VideoEngineInterface : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QQmlListProperty<TrackObjectInfo> listTrackObjectInfos READ listTrackObjectInfos NOTIFY listTrackObjectInfosChanged);
    Q_PROPERTY(QAbstractVideoSurface *videoSurface READ videoSurface WRITE setVideoSurface)
    Q_PROPERTY(QSize sourceSize READ sourceSize NOTIFY sourceSizeChanged)
    Q_PROPERTY(bool enStream READ enStream WRITE setEnStream)
    Q_PROPERTY(bool enSaving READ enSaving WRITE setEnSaving)
    Q_PROPERTY(int sensorMode READ sensorMode WRITE setSensorMode)
    Q_PROPERTY(PlateLog* plateLog READ plateLog WRITE setPlateLog NOTIFY plateLogChanged)
    Q_PROPERTY(int frameID READ frameID)
    Q_PROPERTY(bool enOD READ enOD)
    Q_PROPERTY(bool enTrack READ enTrack)
    Q_PROPERTY(bool enSteer READ enSteer)
public:
    explicit VideoEngineInterface(QObject *parent = nullptr);

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
        return m_frameID;
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

    QMap<int,bool>  freezeMap(){ return m_freezeMap; }
public:
    QAbstractVideoSurface *videoSurface();
    void setVideoSurface(QAbstractVideoSurface *videoSurface);
    QSize sourceSize();
    void update();
    bool allThreadStopped();
public:
    Q_INVOKABLE virtual int addSubViewer(ImageItem *viewer);
    Q_INVOKABLE virtual void removeSubViewer(int viewerID);
    Q_INVOKABLE virtual void setObjectDetect(bool enable){}
    Q_INVOKABLE virtual void setPowerLineDetect(bool enable){}
    Q_INVOKABLE virtual void start(){}
    Q_INVOKABLE virtual void play(){}
    Q_INVOKABLE virtual void stop(){}
    Q_INVOKABLE virtual void setVideo(QString _ip,int _port = 0){}
    Q_INVOKABLE virtual void setStab(bool enable){}
    Q_INVOKABLE virtual void setRecord(bool enable){}
    Q_INVOKABLE virtual void setShare(bool enable){}
    Q_INVOKABLE virtual void setTrackState(bool enable){}
    Q_INVOKABLE virtual void capture(){}
    Q_INVOKABLE virtual void updateFOV(float irFOV, float eoFOV){}
    Q_INVOKABLE virtual void stopTrack(bool enable){}
    Q_INVOKABLE virtual void changeTrackSize(int newSize){}
    Q_INVOKABLE virtual bool getTrackEnable(){}
    Q_INVOKABLE virtual void setStreamMount(QString _streamMount){}
    Q_INVOKABLE virtual void disableObjectDetect(){}
    Q_INVOKABLE virtual void enableObjectDetect(){}
    Q_INVOKABLE virtual void enVisualLock(){}
    Q_INVOKABLE virtual void disVisualLock(){}
    Q_INVOKABLE virtual void setDigitalStab(bool _en){}
    Q_INVOKABLE virtual void setTrackAt(int _id, double _px, double _py, double _w, double _h){}
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
    void determinedPlateOnTracking(QString _imgPath, QString _plateID);
public Q_SLOTS:
    virtual void updateVideoSurface(int width = -1, int height = -1);
    virtual void setTrackType(QString trackType)
    {
        Q_UNUSED(trackType);
    }
    virtual void slDeterminedTrackObjected(int _id, double _px, double _py, double _oW, double _oH, double _w, double _h);
    virtual void slObjectLost();
    virtual void onStreamFrameSizeChanged(int width, int height){}
    virtual void doShowVideo(){}
    virtual void drawOnViewerID(cv::Mat img, int viewerID);
protected:
    virtual PlateLog* plateLog(){return nullptr;}
    virtual void setPlateLog(PlateLog* plateLog){
        Q_UNUSED(plateLog);
    }
protected:
    // === sub viewer
    QList<ImageItem*> m_listSubViewer;
    QMutex imageDataMutex[10];
    unsigned char imageData[10][34041600];
    QMap<int,bool> m_freezeMap;
    // sub viewer ===
    const QVideoFrame::PixelFormat VIDEO_OUTPUT_FORMAT = QVideoFrame::PixelFormat::Format_RGB32;
    QSize m_sourceSize;
    QAbstractVideoSurface *m_videoSurface = nullptr;
    bool m_updateVideoSurface = false;
    int m_updateCount = 0;
    int m_updateMax = 2;
    QSize m_videoSurfaceSize;
    bool m_enStream = true;
    bool m_enSaving = false;
    int m_sensorMode = -1;
    int m_frameID;
    cv::Mat m_imgShow;
    std::string m_logFolder;
    std::string m_logFile;
    VRTSPServer *m_vRTSPServer;
    VSavingWorker *m_vSavingWorker;
    RollBuffer_<rva::GstFrameCacheItem> *m_gstRTSPBuff;
    RollBuffer_<rva::GstFrameCacheItem> *m_buffVideoSaving;

    // OD
    bool m_enSteer = false;
    bool m_enTrack = false;
    bool m_enOD = false;
    bool m_enPD = false;
    QList<TrackObjectInfo *> m_listTrackObjectInfos;
};

#endif // VIDEOENGINEINTERFACE_H
