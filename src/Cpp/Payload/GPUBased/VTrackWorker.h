#ifndef VTRACKWORKER_H
#define VTRACKWORKER_H

#include "Payload/Packet/XPoint.h"
#include <QPoint>
#include <QObject>
#include <QThread>
#include <QWaitCondition>
#include <QRect>
#include <chrono>
#include <time.h>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <string.h>

#include "Utils/Zbar/ZbarLibs.h"
#include "Utils/Files/FileControler.h"
#include "Utils/Files/PlateLog.h"
#include "Payload/Cache/Cache.h"
#include "Payload/Algorithms/stabilizer/dando_02/stab_gcs_kiir.hpp"
#include "Payload/Algorithms/tracker/dando/ITrack.hpp"
#include "Payload/Algorithms/tracker/dando/Utilities.hpp"
#include "Payload/Algorithms/tracker/dando/HTrack/saliency.h"
#include "Payload/Algorithms/tracker/mosse/tracker.h"


#ifdef USE_LINE_DETECTOR
#include <S_PowerLineDetect/power_line_scan.hpp>
#endif
#define MAX_SIZE    640
#define _test_ORBSearcher_
#ifdef _test_ORBSearcher_
#include "../Algorithms/search/ipsearch_orbSearcher.h"
//#define _dbg_show_level_1_
#endif
//#define TRACK_DANDO
//#define DEBUG_TIMER
class ClickTrack;
class GimbalInterface;
class OCR;
class Detector;
using namespace rva;
typedef struct {
    int frameID;
    float panRate;
    float tiltRate;
    float zoomRate;
    float zoomStart;
    float zoomStop;
}joystickData;
class VTrackWorker : public QThread
{
    Q_OBJECT
public:
    explicit VTrackWorker();
    ~VTrackWorker();
    enum KEYPOINT_TYPE{
            NET = 0,
            CONTOURS = 1,
            GOOD_FEATURES = 2,
        };
    void run();
    void pause(bool _pause);
public:
    void moveImage(float panRate,float tiltRate,float zoomRate, float alpha = 0);
    void stop();
    void changeTrackSize(float _trackSize);
    float trackSize(){return m_trackSize;}
    void setClick(float x, float y,float width,float height);
    cv::Mat createPtzMatrix(float w, float h, float dx, float dy,float r,float alpha = 0);
    void setPowerLineDetect(bool enable);
    void setPowerLineDetectRect(QRect rect);
    void setSensorColor(QString colorMode);
private:
    void init();
    void drawObjectBoundary(cv::Mat &_img, cv::Rect _objBoundary,
                            cv::Scalar _color);
    void drawSteeringCenter(cv::Mat &_img, int _wBoundary, int _centerX,
                            const int _centerY, cv::Scalar _color);

Q_SIGNALS:
    void trackStateFound(int _id, double _px, double _py, double _w, double _h, double _oW, double _oH);
    void trackStateLost();
    void determinedPlateOnTracking(QString _imgPath, QString _plateID);
    void zoomCalculateChanged(int index,float zoomCalculate);
    void zoomTargetChanged(float zoomTarget);
    void zoomTargetChangeStopped(float zoomTarget);
private:
    bool m_running = true;
    index_type m_currID;
    RollBuffer<ProcessImageCacheItem> *m_matImageBuff = nullptr;
    RollBuffer<ProcessImageCacheItem> *m_matTrackBuff = nullptr;
    //        RollBuffer<Eye::TrackResponse> *m_rbTrackResIR = nullptr;
    //        RollBuffer<Eye::TrackResponse> *m_rbTrackResEO = nullptr;
    //        RollBuffer<Eye::SystemStatus> *m_rbSystem = nullptr;
    //        RollBuffer<Eye::MotionImage> *m_rbIPCEO = nullptr;
    //        RollBuffer<Eye::MotionImage> *m_rbIPCIR = nullptr;
    const float m_threshold = 0.3f;

    XPoint m_trackPoint;
    std::mutex m_mtx;
    // plate detection
    ClickTrack *m_clickTrack;
public:
    cv::Mat stabMatrix;
    cv::Size m_imgSize;
    PlateLog* m_plateLog;
    QMap<QString,QString> m_mapPlates;
public:
    void setObjDetector(Detector *_detector);
    void setClicktrackDetector(Detector *_detector);
    void setOCR(OCR* _OCR);
    void setDetector(Detector* _detector);
    void startRollbackZoom() {
        for(unsigned int i=0; i< sizeof (m_stopRollBack); i++){
            m_stopRollBack[i] = false;
            m_zoomRateCalculate[i] = 1;
        }
    }
    void createRoiKeypoints(cv::Mat &grayImg,cv::Mat &imgResult,vector<cv::Point2f>& listPoints,
                                KEYPOINT_TYPE type= KEYPOINT_TYPE::NET,int pointPerDimension = 2,int dimensionSize = 200,
                                int dx = 0, int dy = 0);
public:

    // queue joystick command
    GimbalInterface* m_gimbal = nullptr;
    std::deque<joystickData> m_jsQueue;
    QMutex* m_mutexCommand = nullptr;
    QMutex* m_mutex = nullptr;
    QWaitCondition* m_pauseCond = nullptr;
    bool m_stop = false;
    bool m_pause = false;
    int m_frameID;
    int SLEEP_TIME = 1;
    bool m_captureSet = false;
    bool m_startRollbackZoomSet = false;
    bool m_stopRollBack[10];
    cv::Mat m_img;
    cv::Mat m_imgKeyPoints;
    cv::Mat m_i420Img;
    cv::Mat m_grayFrame;
    cv::Mat m_grayFramePrev;
    cv::Mat m_binaryFrame;
    cv::Mat m_imgPrev;
    cv::Mat m_imgDraw;
    cv::Mat m_imgStab;
    cv::Mat m_imgShow;
    cv::Mat m_imgStartTrack;
    vector<cv::Point2f> m_pointsStartOF;
    vector<cv::Point2f> m_pointsPrevOF;
    vector<cv::Point2f> m_pointsCurrentOF;
    int m_pointPerDimension = 8;
    int m_dimensionSize = 400;
    float returnX = 0;
    float returnY = 0;
    bool m_usingIPC = true;
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
    int m_trackSizePrev = 200;
    int m_zoomTimer = 0;
    // click
    bool m_clickSet = false;
    cv::Point2f m_clickPoint;
    float deadZone = 160;
    float maxAxis = 32768.0f;
    // powerline detect
#ifdef USE_LINE_DETECTOR
    bool m_powerLineDetectEnable = false;
    cv::Rect m_powerLineDetectRect;
    my_pli::plr_engine* m_plrEngine = nullptr;
    std::vector<cv::Scalar> m_powerLineList;
    cv::RotatedRect m_plrRR;
#endif
    // color mode
    QMap <QString,int>m_mapColorMode;
    int m_colorMode = 0; // 0: white hot, 1: red hot
    // Object detect
    Detector *m_detector = nullptr;
    int m_objectType = -1;
    // Object search
#ifdef _test_ORBSearcher_
    ip::objsearch::ORBSearcher process;
    int m_maxSearchCount = 300;
    int m_searchCount = 0;
    bool m_objectSearch = false;
#endif
};

#endif // VTRACKWORKER_H