#ifndef VODWORKER_H
#define VODWORKER_H

#include <QObject>
#include <QThread>
#include <chrono>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include "Utils/Zbar/ZbarLibs.h"
#include "Payload/Cache/Cache.h"
// OD
#include "OD/yolo_v2_class.hpp"

using namespace rva;

class VODWorker : public QThread
{
        Q_OBJECT
    public:
        explicit VODWorker();
        ~VODWorker();
        void run();

    public:
        bool isActive();
        void stop();
        void enableOD();
        void disableOD();

        void setDetector(Detector *_detector);

    private:
        index_type m_currID;
        RollBuffer<ProcessImageCacheItem> *m_matImageBuff;
        RollBuffer<DetectedObjectsCacheItem> *m_rbDetectedObjs;
//        RollBuffer<Eye::MotionImage> *m_rbIPCEO;
//        RollBuffer<Eye::MotionImage> *m_rbIPCIR;
        bool m_running = true;
        std::mutex m_mtx;
        std::condition_variable m_cvEnOD;
        bool m_enOD = false;

        Detector *m_detector;
};

#endif // VODWORKER_H