#ifndef VTRACKWORKER_H
#define VTRACKWORKER_H

#include "../Cache/Cache.h"
#include "../ControllerLib/Command/IPCCommands.h"
#include "../ControllerLib/Packet/XPoint.h"
#include "../Zbar/ZbarLibs.h"
#include <QObject>
#include <QThread>
#include <chrono>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <string.h>

#include "../tracker/dando/ITrack.hpp"
#include "../tracker/dando/Utilities.hpp"
#include "../tracker/dando/HTrack/saliency.h"

// plate detection
#include "Clicktrack/clicktrack.h"
#include "OD/yolo_v2_class.hpp"

using namespace rva;

class VTrackWorker : public QThread
{
        Q_OBJECT
    public:
        explicit VTrackWorker();
        ~VTrackWorker();

        void run();

    public:
        void stop();
        void hasNewTrack(int _id, double _px, double _py, double _w, double _h, bool _enSteer, bool _enStab);
        void hasNewMode();
        bool isAcive();
        void changeTrackSize(float _trackSize);
        void disSteer();

    private:
        void init();
        void drawObjectBoundary(cv::Mat &_img, cv::Rect _objBoundary,
                                cv::Scalar _color);
        void drawSteeringCenter(cv::Mat &_img, int _wBoundary, int _centerX,
                                const int _centerY, cv::Scalar _color);

    Q_SIGNALS:
        void determinedTrackObjected(int _id, double _px, double _py, double _w, double _h, double _oW, double _oH);
        void objectLost();
    private:
        bool m_running = true;
        index_type m_currID;
        RollBuffer_<ProcessImageCacheItem> *m_matImageBuff = nullptr;
        RollBuffer<Eye::TrackResponse> *m_rbTrackResIR = nullptr;
        RollBuffer<Eye::TrackResponse> *m_rbTrackResEO = nullptr;
        RollBuffer<Eye::SystemStatus> *m_rbSystem = nullptr;
        RollBuffer<Eye::MotionImage> *m_rbIPCEO = nullptr;
        RollBuffer<Eye::MotionImage> *m_rbIPCIR = nullptr;
        const float m_threshold = 0.3f;

        XPoint m_trackPoint;
        bool m_hasNewTrack = false;
        bool m_hasNewMode = false;
        bool m_trackEn = false;
        bool m_steerEn = false;
        bool m_enStab = false;
        std::mutex m_mtx;
        std::condition_variable m_cvHasNewTrack;

        ITrack *m_tracker;
        int m_trackSize = 100;

        // plate detection
        ClickTrack *m_clickTrack;
public:
        void setClicktrackDetector(Detector *_detector);
};

#endif // VTRACKWORKER_H