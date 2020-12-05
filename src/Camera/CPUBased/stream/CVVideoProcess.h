#ifndef CVVIDEOPROCESS_H
#define CVVIDEOPROCESS_H

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <QObject>
#include <QDebug>
#include <QMutex>
#include <QRect>
#include <QSize>
#include <QVideoFrame>
#include "gstreamer_element.h"
#include <deque>
#include <thread>
#include <chrono>
#include <utility>
#include <map>
#include "../../../Files/PlateLog.h"
#include "../utils/filenameutils.h"
#include "Camera/Algorithms/stabilizer/dando_02/stab_gcs_kiir.hpp"
#include "Camera/Algorithms/tracker/dando/ITrack.hpp"
#include "Camera/Algorithms/tracker/dando/Utilities.hpp"
#include "Camera/Algorithms/tracker/dando/HTrack/saliency.h"
#include "Camera/Algorithms/tracker/mosse/tracker.h"
#include "../../Cache/Cache.h"
#include <chrono>
#define DEBUG_TIME_OFF
//#define TRACK_DANDO
class GimbalInterface;
using namespace rva;
typedef struct {
    int frameID;
    float panRate;
    float tiltRate;
    float zoomRate;
    float zoomStart;
    float zoomStop;
}joystickData;
class  CVVideoProcess : public QObject
{
    Q_OBJECT
public:
    explicit CVVideoProcess(QObject *parent = 0);
    virtual ~CVVideoProcess();

public:
    void moveImage(float panRate,float tiltRate,float zoomRate, float alpha = 0);
    void stop();
    void changeTrackSize(float _trackSize);
    float trackSize(){return m_trackSize;}
    void setClick(float x, float y,float width,float height);
    cv::Mat createPtzMatrix(float w, float h, float dx, float dy,float r,float alpha = 0);
    void pause(bool _pause);
    cv::Point convertPoint(cv::Point originPoint, cv::Mat stabMatrix);
    void capture(bool writeTime = true, bool writeLocation = true);
Q_SIGNALS:
    void processDone();
    void stopped();
    void detectObject();
    void getVideoStopped();
    void trackStateFound(int _id, double _px, double _py, double _oW, double _oH, double _w, double _h);
    void trackStateLost();
    void objectSizeChange(float zoomRate);
    void trackInitSuccess(bool success, int _x, int _y, int _width, int _height);
    void streamFrameSizeChanged(int width, int height);
    void readyDrawOnRenderID(int viewerID, unsigned char *data, int width, int height,float* warpMatrix = nullptr,unsigned char* dataOut = nullptr);
public Q_SLOTS:
    void doWork();
public:
    void msleep(int ms);

//    void warpAffine(cv::Mat &imgY,cv::Mat &imgU,cv::Mat &imgV);
public:
    unsigned char m_renderDataOut[8294400];
    stab_gcs_kiir::vtx_KIIRStabilizer *m_stabilizer;
    cv::Rect object_position;
    std::deque<std::pair<int, GstSample *>> *m_imageQueue;
    cv::Mat *m_imgShow;
    QMutex *m_mutexCapture;
    QMutex *m_mutexProcess;
    // queue joystick command
    GimbalInterface* m_gimbal = nullptr;
    std::deque<joystickData> m_jsQueue;
    QMutex* m_mutexCommand = nullptr;
    QMutex* m_mutex = nullptr;
    QWaitCondition* m_pauseCond = nullptr;
    int *m_frameID;
    const int SLEEP_TIME = 1;
    cv::Mat m_img;
    cv::Mat m_imgI420;
    cv::Mat m_imgI420Warped;
    cv::Mat m_grayFrame;
    cv::Mat m_grayFramePrev;
    QString m_trackType = "sKCF";
    QString m_sensorTrack = "EO";
    float m_irFOV = 0;
    float m_eoFOV = 0;
    bool m_stop = false;
    bool m_pause = false;
    bool m_usingIPC = true;
    bool m_recordEnable = true;
    bool m_sharedEnable = true;
    bool m_setTrack = false;
    std::string m_logFolder;
    std::string m_logFile;
    RollBuffer<GstFrameCacheItem> *m_gstRTSPBuff;
    RollBuffer<GstFrameCacheItem> *m_buffVideoSaving;

    int m_streamWidth = -1;
    int m_streamHeight = - 1;
    PlateLog* m_plateLog;

    // ptz
    float m_dx = -1;
    float m_dy = -1;
    float m_r = 1.0f;
    float m_scale = 1.0f;
    float m_zoomIR = 1.0f;
    float m_zoomDir = 0;
    float m_movePanRate = 0;
    float m_moveTiltRate = 0;
    float m_moveZoomRate = 1;
    float m_rotationAlpha = 0;
    float m_zoomStart = 1.0f;
    float m_zoomRateCalculate[10];
    float m_zoomRateCalculatePrev = 1;
    float m_digitalZoomMax = 20;
    float m_digitalZoomMin = 1;
    int m_countRollBack = 0;
    int m_countRollBackMax = 30;
    cv::Mat m_ptzMatrix;
    std::vector<float> m_warpDataRender;
    // track
#ifdef TRACK_DANDO
    ITrack *m_tracker;
#else
    Tracker *m_tracker;
#endif
    bool m_trackEnable = true;
    bool m_trackSet = false;
    cv::Point m_trackSetPoint;
    cv::Rect m_trackRect;
    cv::Rect m_trackRectPrev;
    // stab
    bool m_stabEnable = true;
    std::string m_stabMode = "STAB_TRACK";
    cv::Mat m_stabMatrix;
    int m_trackSize = 200;
    int m_zoomTimer = 0;
    // click
    bool m_clickSet = false;
    cv::Point2f m_clickPoint;
    float deadZone = 160;
    float maxAxis = 32768.0f;
};

#endif // CVVIDEOPROCESS_H
