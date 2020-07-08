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
#include "Camera/Buffer/RollBuffer.h"
#include "Camera/Buffer/RollBuffer_.h"
#include "../Cache/TrackObject.h"
#include "../Cache/GstFrameCacheItem.h"
#include "../../../Files/PlateLog.h"
#include "../VideoDisplay/VideoRender.h"
#include <opencv2/core.hpp>
#include "Setting/config.h"
class VRTSPServer;
class VSavingWorker;
class VideoRender;
class TrackObjectInfo;
class GimbalInterface;
class Klv{
public:
    Klv(uint8_t key, uint8_t length, std::vector<uint8_t> value){
        m_key = key;
        m_length = length;
        m_value = value;
        encode();
    }
    Klv(uint8_t key, uint8_t length, uint8_t* value){
        m_key = key;
        m_length = length;
        m_value = std::vector<uint8_t>(value,value+length);
        encode();
    }
    ~Klv(){}
    void encode(){
        m_encoded.clear();
        m_encoded.push_back(m_key);
        m_encoded.push_back(m_length);
        m_encoded.insert(m_encoded.end(),m_value.begin(),m_value.end());
    }
public:
    uint8_t m_key;
    uint8_t m_length;
    std::vector<uint8_t> m_value;
    std::vector<uint8_t> m_encoded;
};
class VideoEngine : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QQmlListProperty<TrackObjectInfo> listTrackObjectInfos READ listTrackObjectInfos NOTIFY listTrackObjectInfosChanged);
    Q_PROPERTY(GimbalInterface*    gimbal   READ gimbal       WRITE setGimbal)
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
    explicit VideoEngine(QObject *parent = nullptr);

    QQmlListProperty<TrackObjectInfo> listTrackObjectInfos();
    void addTrackObjectInfo(TrackObjectInfo* object);
    void removeTrackObjectInfo(const int& sequence);
    void removeTrackObjectInfo(const QString &userUid);
    Q_INVOKABLE void updateTrackObjectInfo(const QString& userUid,
                                           const QString& attr,
                                           const QVariant& newValue);
    void setEnStream(bool _enStream);
    bool enStream();
    void setEnSaving(bool _enSaving);
    bool enSaving();
    void setSensorMode(bool _sensorMode);
    bool sensorMode();
    int frameID();
    bool enOD();
    bool enTrack();
    bool enSteer();
    QMap<int,bool> freezeMap();
    GimbalInterface* gimbal();
    virtual void setGimbal(GimbalInterface* gimbal);
public:
    QSize sourceSize();
    void update();
    bool allThreadStopped();
    void loadConfig(Config* config);
    void setSourceRTSP(QString source, int port, int width, int height);
    void stopRTSP();
    virtual void setdigitalZoom(float value){}
    static void drawSteeringCenter(cv::Mat &imgY,cv::Mat &imgU,cv::Mat &imgV,
                                   int _wBoundary, int _centerX, int _centerY,
                                   cv::Scalar _color);
    static void rectangle(cv::Mat& imgY,cv::Mat& imgU,cv::Mat& imgV,cv::Rect rect,cv::Scalar color,
                   int thickness = 1,
                   int lineType = cv::LINE_8, int shift = 0);
    static void fillRectangle(cv::Mat& imgY,cv::Mat& imgU,cv::Mat& imgV,cv::Rect rect,cv::Scalar color);
    static void line(cv::Mat& imgY,cv::Mat& imgU,cv::Mat& imgV,cv::Point start,cv::Point stop,cv::Scalar color,
              int thickness = 1,
              int lineType = cv::LINE_8, int shift = 0);
    static void putText(cv::Mat& imgY,cv::Mat& imgU,cv::Mat& imgV,const string& text, cv::Point org,
                 int fontFace, double fontScale, cv::Scalar color,
                 int thickness = 1, int lineType = cv::LINE_8,
                 bool bottomLeftOrigin = false);
    static void ellipse(cv::Mat& imgY,cv::Mat& imgU,cv::Mat& imgV,cv::Point center,
                 cv::Size size,double angle,double startAngle,double endAngle,
                 cv::Scalar centerColor,int thickness = 1, int lineType = cv::LINE_8, int shift = 0);
    static void convertRGB2YUV(const double R, const double G, const double B, double& Y, double& U, double& V);
    static unsigned short checksum(unsigned char * buff, unsigned short len);
    static std::vector<uint8_t> encodeMeta(GimbalInterface* gimbal);
public:
    Q_INVOKABLE virtual int addVideoRender(VideoRender *viewer);
    Q_INVOKABLE virtual void removeVideoRender(int viewerID);
    Q_INVOKABLE virtual void setObjectDetect(bool enable){}
    Q_INVOKABLE virtual void setPowerLineDetect(bool enable){}
    Q_INVOKABLE virtual void setPowerLineDetectRect(QRect rect){}
    Q_INVOKABLE virtual void start(){}
    Q_INVOKABLE virtual void play(){}
    Q_INVOKABLE virtual void stop(){}
    Q_INVOKABLE virtual void setVideo(QString _ip,int _port = 0){}
    Q_INVOKABLE virtual void setStab(bool enable){}
    Q_INVOKABLE virtual void setRecord(bool enable){}
    Q_INVOKABLE virtual void setSensorColor(QString colorMode){}
    Q_INVOKABLE virtual void setShare(bool enable){}
    Q_INVOKABLE virtual void setTrackState(bool enable){}
    Q_INVOKABLE virtual void capture(){}
    Q_INVOKABLE virtual void updateFOV(float irFOV, float eoFOV){}
    Q_INVOKABLE virtual void changeTrackSize(int newSize){}
    Q_INVOKABLE virtual bool getTrackEnable(){}
    Q_INVOKABLE virtual void setStreamMount(QString _streamMount){}
    Q_INVOKABLE virtual void disableObjectDetect(){}
    Q_INVOKABLE virtual void enableObjectDetect(){}
    Q_INVOKABLE virtual void moveImage(float panRate,float tiltRate,float zoomRate,float alpha = 0){}
    Q_INVOKABLE virtual void setDigitalStab(bool _en){}
    Q_INVOKABLE virtual void setTrackAt(int _id, double _px, double _py, double _w, double _h){}
    Q_INVOKABLE virtual void pause(bool pause){}
    Q_INVOKABLE virtual void goToPosition(float percent){}
    Q_INVOKABLE virtual void setSpeed(float speed){}
    Q_INVOKABLE virtual qint64 getTime(QString type){return 0;}
    Q_INVOKABLE virtual void setObjectSearch(bool enable){}
Q_SIGNALS:
    void sourceLinkChanged(bool isVideo);
    void listTrackObjectInfosChanged();
    void sourceSizeChanged(int newWidth, int newHeight);
    void readyToRead();
    void trackStateFound(int _id, double _px, double _py, double _oW, double _oH, double _w, double _h);
    void trackStateLost();
    void needZoomChange(float deltaFOV);
    void started();
    void stopped();
    void objectDetected();
    void trackInitSuccess(bool success, int _x, int _y, int _width, int _height);
    void plateLogChanged();
    void determinedPlateOnTracking(QString _imgPath, QString _plateID);
public Q_SLOTS:
    virtual void setTrackType(QString trackType)
    {
        Q_UNUSED(trackType);
    }
    virtual void slTrackStateFound(int _id, double _px, double _py, double _oW, double _oH, double _w, double _h);
    virtual void slTrackStateLost();
    virtual void onStreamFrameSizeChanged(int width, int height);
    virtual void doShowVideo(){}
    virtual void drawOnRenderID(int viewerID, unsigned char *data, int width, int height,
                        float* warpMatrix = nullptr, unsigned char *dataOut = nullptr);

protected:
    virtual PlateLog* plateLog(){return nullptr;}
    virtual void setPlateLog(PlateLog* plateLog){
        Q_UNUSED(plateLog);
    }

protected:
    GimbalInterface* m_gimbal = nullptr;
    // === sub viewer
    QList<VideoRender*> m_listRender;
    QMutex imageDataMutex[10];
    unsigned char imageData[10][34041600];
    QMap<int,bool> m_freezeMap;
    // sub viewer ===
    QSize m_sourceSize;
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
    VRTSPServer *m_vRTSPServer = nullptr;
    VSavingWorker *m_vSavingWorker = nullptr;

    // OD
    bool m_enSteer = false;
    bool m_enTrack = false;
    bool m_enOD = false;
    bool m_enPD = false;
    QList<TrackObjectInfo *> m_listTrackObjectInfos;

    // Config
    Config* m_config = nullptr;
};

#endif // VIDEOENGINEINTERFACE_H
