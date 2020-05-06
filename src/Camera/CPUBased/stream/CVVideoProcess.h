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
#include "../stabilizer/dando_02/stab_gcs_kiir.hpp"
#include "../tracker/dando/thresholding.hpp"
#include "../tracker/dando/SKCF/skcf.h"
#include "../tracker/dando/ITrack.hpp"
#include "../tracker/dando/Utilities.hpp"
#include "../tracker/dando/HTrack/saliency.h"
#include "../detector/movingObject.hpp"
#include "../tracker/dando/InitTracking.hpp"
#include "../../Cache/Cache.h"
#include <chrono>
#define MOVE_CLICK_TRACK
//#define USE_SALIENCY
#define DEBUG_TIME_OFF
using namespace rva;
class  CVVideoProcess : public QObject
{
        Q_OBJECT
    public:
        explicit CVVideoProcess(QObject *parent = 0);
        virtual ~CVVideoProcess();
        void setTrackType(QString trackType);
    public:
        stab_gcs_kiir::vtx_KIIRStabilizer *m_stabilizer;
        ThresholdingTracker *thresh_tracker;
        InitTracking *ClickTrackObj;
        KTrackers *k_tracker;
        ITrack *m_tracker;
        MovingDetector *m_detector;
        cv::Rect object_position;
        std::deque<std::pair<int, GstSample *>> *m_imageQueue;
        cv::Mat *m_imgShow;
        QMutex *m_mutexCapture;
        QMutex *m_mutexProcess;
        //    cv::Mat m_imgRaw;
        int *m_frameID;
        const int SLEEP_TIME = 1;
        cv::Mat m_img;
        cv::Mat m_imgStab;
        cv::Mat m_imgTrack;
        cv::Mat grayFrame;
        cv::Mat enhancedFrame;
        cv::Mat m_img_get;
        QString m_trackType = "sKCF";
        QString m_sensorTrack = "EO";
        float m_irFOV = 0;
        float m_eoFOV = 0;
        bool m_stop = false;
        bool m_usingIPC = true;
        bool m_trackEnable = true;
        bool m_stabEnable = false;
        bool m_recordEnable = false;
        bool m_sharedEnable = true;
        bool m_setTrack = false;
        bool GlobalTrackInited;
        cv::Point m_pointSetTrack;
        float m_cropRatio = 1.0f;
        int m_trackSize = 50;
        bool m_detectEnable = false;
        int m_detectSize = 100;
        int m_objectArea = 50;
        int m_zoomTimer = 0;

        std::string m_logFolder;
        std::string m_logFile;
        RollBuffer_<GstFrameCacheItem> *m_gstRTSPBuff;
        RollBuffer_<GstFrameCacheItem> *m_buffVideoSaving;

        int m_streamWidth = -1;
        int m_streamHeight = - 1;
        PlateLog* m_plateLog;
    public:
        void setTrack(int x, int y);
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
        void readyDrawOnViewerID(cv::Mat img, int viewerID);
    public Q_SLOTS:
        void capture();
        void doWork();
    public:
        void msleep(int ms);
};

#endif // CVVIDEOPROCESS_H
